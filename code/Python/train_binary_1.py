#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline binario completo (versión corregida)

Incluye:
1) Carga y limpieza de datos
2) Ranking univariante global (solo exploratorio)
3) Selección de variables TRAIN-ONLY por fold:
   - screening univariante
   - filtro de correlación
   - RFECV con Logistic L1 y CV por grupos
4) Repeated Stratified Group Cross-Validation
5) Comparación estadística de modelos
6) Selección del mejor modelo
7) Lanzamiento del Script 3 corregido para modelo final / calibración / SHAP
"""

import argparse
import ast
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science", "grid"])
plt.rcParams["text.usetex"] = False
dpi = 300

from scipy.stats import shapiro, mannwhitneyu, ttest_ind

from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFECV

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix
)


# ==============================================================================
# UTILIDADES GENERALES
# ==============================================================================

def save_plot_both_formats(fig_path_base, dpi=300, bbox_inches="tight"):
    """
    Guarda una figura en PNG y PDF.
    """
    png_path = f"{fig_path_base}.png"
    pdf_path = f"{fig_path_base}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(pdf_path, format="pdf", bbox_inches=bbox_inches)
    print(f"  --> Guardado: {png_path}")
    print(f"  --> Guardado: {pdf_path}")


def safe_shapiro(x, random_state=42, max_n=500):
    """
    Shapiro-Wilk robusto para n grandes.
    """
    x = pd.Series(x).dropna()

    if len(x) < 3 or x.nunique(dropna=True) < 2:
        return 1.0

    if len(x) > max_n:
        x = x.sample(max_n, random_state=random_state)

    try:
        _, p = shapiro(x)
    except Exception:
        p = 1.0

    return p


def compute_empirical_ci(values, alpha=0.05):
    """
    IC empírico por percentiles.
    """
    values = pd.Series(values).dropna().values
    if len(values) == 0:
        return np.nan, np.nan
    low = np.percentile(values, 100 * (alpha / 2))
    high = np.percentile(values, 100 * (1 - alpha / 2))
    return low, high


def summarize_results_with_ci(df_results, outpath):
    """
    Resume métricas por clasificador con media, std e IC empírico.
    """
    metrics_to_summarize = [
        "val_auc",
        "val_sensitivity",
        "val_specificity",
        "val_f1_binary",
        "val_balanced_accuracy",
        "val_mcc",
        "n_features"
    ]

    rows = []
    for clf in sorted(df_results["Classifier"].unique()):
        df_clf = df_results[df_results["Classifier"] == clf]

        row = {"Classifier": clf}
        for metric_col in metrics_to_summarize:
            vals = df_clf[metric_col].dropna().values
            if len(vals) == 0:
                row[f"{metric_col}_mean"] = np.nan
                row[f"{metric_col}_std"] = np.nan
                row[f"{metric_col}_ci_low"] = np.nan
                row[f"{metric_col}_ci_high"] = np.nan
            else:
                row[f"{metric_col}_mean"] = np.mean(vals)
                row[f"{metric_col}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                ci_low, ci_high = compute_empirical_ci(vals)
                row[f"{metric_col}_ci_low"] = ci_low
                row[f"{metric_col}_ci_high"] = ci_high

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(outpath, index=False)
    print(f">> Resumen con IC guardado en: {outpath}")
    return df_summary


def get_continuous_scores(fitted_model, X):
    """
    Devuelve un score continuo:
    - predict_proba[:,1] si existe
    - decision_function si no
    - None si no hay ninguno
    """
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]
    elif hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X)
    else:
        return None


def choose_optimal_threshold(y_true, scores):
    """
    Elige el threshold óptimo (Youden) usando SOLO train.
    Sirve tanto para probabilidades como para decision scores.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return 0.5

    if len(np.unique(scores)) < 2:
        return float(np.median(scores))

    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, scores, pos_label=1)
        youden = tpr - fpr
        best_idx = int(np.nanargmax(youden))
        best_threshold = thresholds[best_idx]

        if not np.isfinite(best_threshold):
            best_threshold = float(np.median(scores))

        return float(best_threshold)
    except Exception:
        return float(np.median(scores))


# ==============================================================================
# MODELOS
# ==============================================================================

def get_models(random_state=42):
    """
    Define pipelines para cada clasificador.
    """
    pipe_svc = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        SVC(random_state=random_state, class_weight="balanced", probability=True)
    )

    pipe_lr = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            class_weight="balanced",
            random_state=random_state,
            solver="saga",
            max_iter=10000
        )
    )

    pipe_rf = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state
        )
    )

    pipe_nb = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        GaussianNB()
    )

    pipe_knn = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        KNeighborsClassifier(n_jobs=-1)
    )

    pipe_gb = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        GradientBoostingClassifier(random_state=random_state)
    )

    models = [
        ("SVM", pipe_svc),
        ("Logistic Regression", pipe_lr),
        ("Random Forest", pipe_rf),
        ("Naive Bayes", pipe_nb),
        ("KNN", pipe_knn),
        ("Gradient Boosting", pipe_gb),
    ]
    return models


# ==============================================================================
# BLOQUE 2: RANKING GLOBAL EXPLORATORIO
# ==============================================================================

def generate_global_univariate_ranking(X, y, fs_dir, images_top_k=20, random_state=42):
    """
    Ranking univariante global SOLO exploratorio.
    No se usa para entrenar.
    """
    os.makedirs(fs_dir, exist_ok=True)
    images_dir = os.path.join(fs_dir, "images_global_ranking_1")
    os.makedirs(images_dir, exist_ok=True)

    ranking_rows = []

    for col in X.columns:
        x_col = X[col]

        if pd.Series(x_col).nunique(dropna=True) < 2:
            ranking_rows.append({
                "Feature": col,
                "AUC": 0.5,
                "Cutoff": np.nan,
                "Sensitivity": np.nan,
                "Specificity": np.nan,
                "Test": "constant",
                "p-value": 1.0
            })
            continue

        p_norm = safe_shapiro(x_col, random_state=random_state)
        a = x_col[y == 0]
        b = x_col[y == 1]

        try:
            if p_norm > 0.05:
                _, pval = ttest_ind(a, b, equal_var=False, nan_policy="omit")
                test_name = "t-test"
            else:
                _, pval = mannwhitneyu(a, b, alternative="two-sided")
                test_name = "mann-whitney"
        except Exception:
            pval = 1.0
            test_name = "failed"

        try:
            fpr, tpr, thresholds = metrics.roc_curve(y, x_col, pos_label=1)
            auc_val = metrics.auc(fpr, tpr)

            if auc_val < 0.5:
                fpr, tpr, thresholds = metrics.roc_curve(y, x_col, pos_label=0)
                auc_val = metrics.auc(fpr, tpr)

            youden = tpr - fpr
            best_idx = int(np.nanargmax(youden))

            cutoff = thresholds[best_idx]
            sens = tpr[best_idx]
            spec = 1 - fpr[best_idx]
        except Exception:
            auc_val = 0.5
            cutoff = np.nan
            sens = np.nan
            spec = np.nan

        ranking_rows.append({
            "Feature": col,
            "AUC": auc_val,
            "Cutoff": cutoff,
            "Sensitivity": sens,
            "Specificity": spec,
            "Test": test_name,
            "p-value": pval
        })

    df_ranking = pd.DataFrame(ranking_rows).set_index("Feature").sort_values(by="p-value")
    ranking_path = os.path.join(fs_dir, "ranking_univariante_global.csv")
    df_ranking.to_csv(ranking_path)
    print(f">> Ranking univariante global guardado en: {ranking_path}")

    # Violin plots para las top globales (exploratorio)
    top_features = df_ranking.index[:images_top_k].tolist()
    for rank, feature_name in enumerate(top_features, start=1):
        try:
            safe_feat_name = str(feature_name).replace("/", "_")
            feat_dir = os.path.join(images_dir, f"{rank:02d}_{safe_feat_name}")
            os.makedirs(feat_dir, exist_ok=True)

            plt.figure(figsize=(8, 6))
            sns.violinplot(x=y, y=X[feature_name], color="grey")
            plt.title(f"Distribución global: {feature_name}")
            plt.xlabel("Clase")
            plt.ylabel(feature_name)
            plt.tight_layout()
            save_plot_both_formats(os.path.join(feat_dir, f"{safe_feat_name}_violinplot"), dpi=dpi)
            plt.close()
        except Exception as e:
            print(f"[Aviso] No se pudo generar violinplot para {feature_name}: {e}")

    # Correlación global de las top 20 (exploratoria)
    try:
        top_for_corr = [f for f in top_features if f in X.columns][:20]
        if len(top_for_corr) > 1:
            corr_matrix = X[top_for_corr].corr(method="spearman")
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
            plt.title("Correlación global (top features del ranking)")
            plt.tight_layout()
            save_plot_both_formats(os.path.join(fs_dir, "global_top_features_correlation"), dpi=dpi)
            plt.close()
    except Exception as e:
        print(f"[Aviso] No se pudo generar la matriz de correlación global: {e}")

    return df_ranking


# ==============================================================================
# SELECCIÓN TRAIN-ONLY POR FOLD
# ==============================================================================

def select_features_train_only(
    X_train,
    y_train,
    groups_train,
    corr_threshold=0.85,
    min_features=2,
    random_state=42
):
    """
    Selección de variables SOLO con TRAIN:
      1) Screening univariante (p-value)
      2) Filtro de redundancia (Spearman)
      3) RFECV multivariante con Logistic L1 y CV por grupos
    """
    X_work = X_train.copy()

    # ---------- 1) Ranking univariante TRAIN-ONLY ----------
    pvals = {}
    aucs = {}

    for col in X_work.columns:
        x_col = X_work[col]

        if pd.Series(x_col).nunique(dropna=True) < 2:
            pvals[col] = 1.0
            aucs[col] = 0.5
            continue

        p_norm = safe_shapiro(x_col, random_state=random_state)

        a = x_col[y_train == 0]
        b = x_col[y_train == 1]

        try:
            if p_norm > 0.05:
                _, pval = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            else:
                _, pval = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            pval = 1.0

        try:
            fpr, tpr, _ = metrics.roc_curve(y_train, x_col, pos_label=1)
            auc_val = np.trapz(tpr, fpr)
            if auc_val < 0.5:
                fpr, tpr, _ = metrics.roc_curve(y_train, x_col, pos_label=0)
                auc_val = np.trapz(tpr, fpr)
        except Exception:
            auc_val = 0.5

        pvals[col] = pval
        aucs[col] = auc_val

    ranked_cols = sorted(pvals.keys(), key=lambda c: (pvals[c], -aucs[c]))
    X_ranked = X_work[ranked_cols]

    # ---------- 2) Filtro de correlación ----------
    corr = X_ranked.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X_clean = X_ranked.drop(columns=to_drop, errors="ignore")

    if X_clean.shape[1] < min_features:
        return ranked_cols[:min_features]

    # ---------- 3) RFECV multivariante con grupos ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    base_model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        C=1.0,
        max_iter=5000,
        random_state=random_state
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    selector = RFECV(
        estimator=base_model,
        scoring="roc_auc",
        cv=inner_cv,
        step=1,
        min_features_to_select=min_features,
        n_jobs=-1
    )

    selector.fit(X_scaled, y_train, groups=groups_train)

    selected_features = X_clean.columns[selector.support_].tolist()

    if len(selected_features) < min_features:
        selected_features = ranked_cols[:min_features]

    return selected_features


# ==============================================================================
# CV REPETIDA POR GRUPOS
# ==============================================================================

def run_repeated_group_cv(models, X, y, groups, n_splits=5, n_repeats=10, base_random_state=42):
    """
    Ejecuta repeated StratifiedGroupKFold para todos los modelos,
    con selección TRAIN-ONLY por fold.
    """
    all_results = []
    all_predictions = []
    all_selected_features = []

    # Precomputar particiones para asegurar que todos los modelos usan EXACTAMENTE
    # los mismos folds, con IDs consistentes.
    split_specs = []
    for rep in range(n_repeats):
        current_random_state = base_random_state + rep
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=current_random_state
        )

        for fold_in_repeat, (train_idx, val_idx) in enumerate(
            splitter.split(X, y, groups=groups), start=1
        ):
            global_fold_id = rep * n_splits + fold_in_repeat
            split_specs.append({
                "Repeat": rep + 1,
                "Fold": global_fold_id,
                "Fold_in_repeat": fold_in_repeat,
                "train_idx": train_idx,
                "val_idx": val_idx
            })

    for model_name, base_model in models:
        print(f"\nEvaluating {model_name}...")

        for spec in split_specs:
            rep = spec["Repeat"]
            fold_id = spec["Fold"]
            fold_in_repeat = spec["Fold_in_repeat"]
            train_idx = spec["train_idx"]
            val_idx = spec["val_idx"]

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]

            # ---------------------------
            # SELECCIÓN DE VARIABLES TRAIN-ONLY
            # ---------------------------
            selected_features = select_features_train_only(
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                corr_threshold=0.85,
                min_features=2,
                random_state=base_random_state + rep
            )

            X_train_sel = X_train[selected_features].copy()
            X_val_sel = X_val[selected_features].copy()

            all_selected_features.append({
                "Classifier": model_name,
                "Repeat": rep,
                "Fold": fold_id,
                "Fold_in_repeat": fold_in_repeat,
                "Selected_Features": selected_features
            })

            # ---------------------------
            # ENTRENAMIENTO
            # ---------------------------
            model = clone(base_model)
            model.fit(X_train_sel, y_train)

            # Scores continuos
            y_train_score = get_continuous_scores(model, X_train_sel)
            y_val_score = get_continuous_scores(model, X_val_sel)

            # Threshold óptimo aprendido SOLO en train
            if y_train_score is not None:
                best_threshold = choose_optimal_threshold(y_train, y_train_score)
                y_train_pred = (np.asarray(y_train_score) >= best_threshold).astype(int)
                y_val_pred = (np.asarray(y_val_score) >= best_threshold).astype(int) if y_val_score is not None else model.predict(X_val_sel)
            else:
                best_threshold = np.nan
                y_train_pred = model.predict(X_train_sel)
                y_val_pred = model.predict(X_val_sel)

            # ---------------------------
            # MÉTRICAS TRAIN
            # ---------------------------
            try:
                train_auc = roc_auc_score(y_train, y_train_score) if y_train_score is not None else np.nan
            except Exception:
                train_auc = np.nan

            train_f1 = f1_score(y_train, y_train_pred, average="binary", zero_division=0)

            # ---------------------------
            # MÉTRICAS VALIDACIÓN
            # ---------------------------
            try:
                val_auc = roc_auc_score(y_val, y_val_score) if y_val_score is not None else np.nan
            except Exception:
                val_auc = np.nan

            cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])

            val_mcc = matthews_corrcoef(y_val, y_val_pred)
            val_kappa = cohen_kappa_score(y_val, y_val_pred)
            val_f1_binary = f1_score(y_val, y_val_pred, average="binary", zero_division=0)
            val_f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
            val_sensitivity = recall_score(y_val, y_val_pred, pos_label=1, zero_division=0)
            val_specificity = recall_score(y_val, y_val_pred, pos_label=0, zero_division=0)
            val_ppv = precision_score(y_val, y_val_pred, pos_label=1, zero_division=0)

            # NPV = TN / (TN + FN)
            denom_npv = cm[0, 0] + cm[1, 0]
            val_npv = cm[0, 0] / denom_npv if denom_npv > 0 else np.nan

            per_class_precision = precision_score(y_val, y_val_pred, average=None, labels=[0, 1], zero_division=0)
            per_class_recall = recall_score(y_val, y_val_pred, average=None, labels=[0, 1], zero_division=0)
            per_class_f1 = f1_score(y_val, y_val_pred, average=None, labels=[0, 1], zero_division=0)

            per_class_accuracy = []
            for i in range(len(cm)):
                row_sum = np.sum(cm[i, :])
                if row_sum > 0:
                    per_class_accuracy.append(cm[i, i] / row_sum)
                else:
                    per_class_accuracy.append(np.nan)

            all_results.append({
                "Classifier": model_name,
                "Repeat": rep,
                "Fold": fold_id,
                "Fold_in_repeat": fold_in_repeat,
                "train_auc": train_auc,
                "train_f1": train_f1,
                "best_threshold": best_threshold,
                "val_auc": val_auc,
                "val_mcc": val_mcc,
                "val_kappa": val_kappa,
                "val_f1_binary": val_f1_binary,
                "val_f1_macro": val_f1_macro,
                "val_accuracy": val_accuracy,
                "val_sensitivity": val_sensitivity,
                "val_specificity": val_specificity,
                "val_ppv": val_ppv,
                "val_npv": val_npv,
                "val_balanced_accuracy": val_balanced_accuracy,
                "per_class_precision": per_class_precision.tolist(),
                "per_class_recall": per_class_recall.tolist(),
                "per_class_f1": per_class_f1.tolist(),
                "per_class_accuracy": per_class_accuracy,
                "n_features": len(selected_features)
            })

            all_predictions.append({
                "Classifier": model_name,
                "Repeat": rep,
                "Fold": fold_id,
                "Fold_in_repeat": fold_in_repeat,
                "y_val": y_val.tolist(),
                "y_pred": y_val_pred.tolist(),
                "y_prob": y_val_score.tolist() if y_val_score is not None else [],
                "best_threshold": best_threshold
            })

    df_results = pd.DataFrame(all_results)
    df_preds = pd.DataFrame(all_predictions)
    df_features = pd.DataFrame(all_selected_features)

    # Ordenar
    sort_cols = ["Classifier", "Repeat", "Fold"]
    df_results.sort_values(by=sort_cols, inplace=True)
    df_preds.sort_values(by=sort_cols, inplace=True)
    df_features.sort_values(by=sort_cols, inplace=True)

    return df_results, df_preds, df_features


# ==============================================================================
# ROC CURVES
# ==============================================================================

def generate_roc_plots(df_results, df_preds, outdir):
    """
    Genera ROC del fold óptimo y del fold mediano por clasificador.
    """
    roc_dir = outdir
    os.makedirs(roc_dir, exist_ok=True)

    roc_plot_path_opt = os.path.join(roc_dir, "roc_optimal_folds")
    roc_plot_path_med = os.path.join(roc_dir, "roc_median_folds")

    curves_info_optimal = []
    curves_info_median = []

    classifiers = df_results["Classifier"].unique()

    for clf_name in classifiers:
        df_clf = df_results[df_results["Classifier"] == clf_name].copy()

        # Mejor fold por AUC
        best_fold_idx = df_clf["val_auc"].idxmax()
        best_fold_num = df_clf.loc[best_fold_idx, "Fold"]

        # Fold con AUC más cercana a la mediana
        median_auc = df_clf["val_auc"].median()
        median_fold_idx = (df_clf["val_auc"] - median_auc).abs().idxmin()
        median_fold_num = df_clf.loc[median_fold_idx, "Fold"]

        # Fold óptimo
        df_best = df_preds[
            (df_preds["Classifier"] == clf_name) &
            (df_preds["Fold"] == best_fold_num)
        ]

        if len(df_best) > 0:
            y_val_list = df_best.iloc[0]["y_val"]
            y_prob_list = df_best.iloc[0]["y_prob"]

            if isinstance(y_prob_list, list) and len(y_prob_list) > 0:
                fpr, tpr, _ = metrics.roc_curve(y_val_list, y_prob_list, pos_label=1)
                auc_val = metrics.auc(fpr, tpr)
                curves_info_optimal.append({
                    "classifier": clf_name,
                    "fold": best_fold_num,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": auc_val
                })

        # Fold mediano
        df_median = df_preds[
            (df_preds["Classifier"] == clf_name) &
            (df_preds["Fold"] == median_fold_num)
        ]

        if len(df_median) > 0:
            y_val_list = df_median.iloc[0]["y_val"]
            y_prob_list = df_median.iloc[0]["y_prob"]

            if isinstance(y_prob_list, list) and len(y_prob_list) > 0:
                fpr, tpr, _ = metrics.roc_curve(y_val_list, y_prob_list, pos_label=1)
                auc_val = metrics.auc(fpr, tpr)
                curves_info_median.append({
                    "classifier": clf_name,
                    "fold": median_fold_num,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": auc_val
                })

    curves_info_optimal.sort(key=lambda x: x["auc"], reverse=True)
    curves_info_median.sort(key=lambda x: x["auc"], reverse=True)

    my_colors = ["#0072B2", "#009E73", "#D55E00", "#CC78BC", "#DE8F05", "#56B4E9"]
    my_palette = sns.color_palette(my_colors)
    fixed_classifiers = ["SVM", "Logistic Regression", "Random Forest", "Naive Bayes", "KNN", "Gradient Boosting"]
    color_mapping = {clf: my_palette[i] for i, clf in enumerate(fixed_classifiers)}

    # ROC óptima
    fig_opt, ax_opt = plt.subplots(figsize=(8, 6))
    for info in curves_info_optimal:
        ax_opt.plot(
            info["fpr"],
            info["tpr"],
            label=f"{info['classifier']} (Fold={info['fold']}, AUC={info['auc']:.3f})",
            color=color_mapping.get(info["classifier"], None)
        )

    ax_opt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="_nolegend_")
    ax_opt.set_xlabel("False Positive Rate")
    ax_opt.set_ylabel("True Positive Rate")
    ax_opt.legend(fontsize=9)
    fig_opt.tight_layout()
    save_plot_both_formats(roc_plot_path_opt, dpi=dpi)
    plt.close(fig_opt)

    # ROC mediana
    fig_med, ax_med = plt.subplots(figsize=(8, 6))
    for info in curves_info_median:
        ax_med.plot(
            info["fpr"],
            info["tpr"],
            label=f"{info['classifier']} (Fold={info['fold']}, AUC={info['auc']:.3f})",
            color=color_mapping.get(info["classifier"], None)
        )

    ax_med.plot([0, 1], [0, 1], linestyle="--", color="gray", label="_nolegend_")
    ax_med.set_xlabel("False Positive Rate")
    ax_med.set_ylabel("True Positive Rate")
    ax_med.legend(fontsize=9)
    fig_med.tight_layout()
    save_plot_both_formats(roc_plot_path_med, dpi=dpi)
    plt.close(fig_med)

    print(f">> Curvas ROC guardadas en: {roc_dir}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatización binaria completa (versión corregida)")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--results_base",
        type=str,
        default="/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/data/binary_new"
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument(
        "--resume_run",
        type=str,
        default=None,
        help="Nombre de carpeta de un run previo para reutilizar resultados."
    )
    args = parser.parse_args()

    # ==========================================================================
    # BLOQUE 1: GESTIÓN DE RUN
    # ==========================================================================

    if args.resume_run:
        experiment_dir = os.path.join(args.results_base, "runs_1", f"{args.resume_run}_1")
        print(f">>> Reutilizando/derivando experimento en: {experiment_dir}")
        os.makedirs(experiment_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(args.results_base, "runs_1", f"run_bin_{timestamp}_1")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f">>> Iniciando nuevo experimento en: {experiment_dir}")

    fs_dir = os.path.join(experiment_dir, "feature_selection_1")
    roc_dir = os.path.join(experiment_dir, "ROC_curves_1")

    results_filepath = os.path.join(experiment_dir, "results.csv")
    preds_filepath = os.path.join(experiment_dir, "predictions.csv")
    features_filepath = os.path.join(experiment_dir, "features_per_fold.csv")
    summary_with_ci_path = os.path.join(experiment_dir, "metrics_with_ci.csv")

    # ==========================================================================
    # BLOQUE 2: CARGA + LIMPIEZA + RANKING GLOBAL EXPLORATORIO
    # ==========================================================================

    df = pd.read_csv(args.csv)

    if "label" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'label'.")
    if "patient_id" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'patient_id'.")

    y = df["label"].values
    groups = df["patient_id"].values

    cols_to_drop = ["id_igtp", "patient_id", "study_id", "label", "mask_type", "SSA_type"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if "diagnostics" in c], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    print(f">> Dataset cargado: {X.shape[0]} muestras | {X.shape[1]} variables numéricas")

    # Ranking exploratorio global
    generate_global_univariate_ranking(X, y, fs_dir=fs_dir, images_top_k=20, random_state=42)

    # ==========================================================================
    # BLOQUE 3: ENTRENAMIENTO Y EVALUACIÓN (REPEATED CV + FEATURE SELECTION)
    # ==========================================================================

    if os.path.exists(results_filepath) and os.path.exists(preds_filepath) and os.path.exists(features_filepath):
        print(f"\n>>> Resultados encontrados en {experiment_dir}. Cargando...")
        df_results = pd.read_csv(results_filepath)
        df_preds = pd.read_csv(
            preds_filepath,
            converters={
                "y_val": ast.literal_eval,
                "y_pred": ast.literal_eval,
                "y_prob": ast.literal_eval
            }
        )
        df_features = pd.read_csv(
            features_filepath,
            converters={"Selected_Features": ast.literal_eval}
        )
    else:
        print("\n>>> Iniciando entrenamiento con repeated CV y selección por fold...")
        models = get_models(random_state=42)

        df_results, df_preds, df_features = run_repeated_group_cv(
            models=models,
            X=X,
            y=y,
            groups=groups,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            base_random_state=42
        )

        df_results.to_csv(results_filepath, index=False)
        df_preds.to_csv(preds_filepath, index=False)
        df_features.to_csv(features_filepath, index=False)

        print(f">> Results guardado en: {results_filepath}")
        print(f">> Predictions guardado en: {preds_filepath}")
        print(f">> Features por fold guardado en: {features_filepath}")

    # Resumen con IC
    summarize_results_with_ci(df_results, summary_with_ci_path)

    # Curvas ROC
    roc_opt_png = os.path.join(roc_dir, "roc_optimal_folds.png")
    roc_med_png = os.path.join(roc_dir, "roc_median_folds.png")
    if os.path.exists(roc_opt_png) and os.path.exists(roc_med_png):
        print(f">>> Curvas ROC ya existentes en {roc_dir}. Saltando...")
    else:
        generate_roc_plots(df_results, df_preds, roc_dir)

    # ==========================================================================
    # BLOQUE 4: COMPARACIÓN ESTADÍSTICA DE MODELOS
    # ==========================================================================

    model_diff_dir = os.path.join(experiment_dir, "model_differences_1")
    summary_txt = os.path.join(model_diff_dir, "model_differences_summary.txt")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.exists(summary_txt):
        print(f"\n>>> Comparación estadística ya existente en {model_diff_dir}. Saltando...")
    else:
        print("\nExecuting model comparisons (2_model_differences.py)...")
        os.makedirs(model_diff_dir, exist_ok=True)

        postprocess_cmd = [
            "python3",
            os.path.join(script_dir, "../2_model_differences.py"),
            "--csv_preds", preds_filepath,
            "--csv_results", results_filepath,
            "--metric", "val_auc",
            "--alpha", "0.05",
            "--outdir", model_diff_dir
        ]

        try:
            subprocess.run(postprocess_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("\n[ERROR] Falló el script de comparación estadística.")
            print(f"Comando ejecutado: {' '.join(postprocess_cmd)}")
            print(f"Código de salida: {e.returncode}")
            raise

    # ==========================================================================
    # BLOQUE 5: OPTIMIZACIÓN FINAL Y EXPLICABILIDAD (SCRIPT 3 CORREGIDO)
    # ==========================================================================

    best_results_dir = os.path.join(experiment_dir, "best_results_1")
    features_per_fold_path = features_filepath

    if os.path.exists(best_results_dir):
        print(f"\n>>> El re-entrenamiento del mejor modelo ya existe en {best_results_dir}. Saltando...")
    else:
        print("\n>>> Seleccionando el modelo ganador (equilibrio rendimiento / estabilidad)...")

        means = df_results.groupby("Classifier")["val_auc"].mean()
        stds = df_results.groupby("Classifier")["val_auc"].std()

        quality_score = means - stds
        best_model = quality_score.idxmax()

        print(f"GANADOR SELECCIONADO: {best_model}")
        print(f"  --> Score de calidad (Mean - Std): {quality_score[best_model]:.3f}")
        print(f"  --> Mean AUC: {means[best_model]:.3f}")
        print(f"  --> Std AUC: {stds[best_model]:.3f}")
        
        model_mapping = {
            "SVM": "SVM",
            "Logistic Regression": "LogisticRegression",
            "Random Forest": "RandomForest",
            "Naive Bayes": "NaiveBayes",
            "KNN": "KNN",
            "Gradient Boosting": "GradientBoosting"
        }

        best_model_finetune = model_mapping.get(best_model, best_model)

        print(f"\n>>> Lanzando fine-tuning del mejor modelo: {best_model_finetune}")

        fine_tune_cmd = [
            "python3",
            os.path.join(script_dir, "3_retrain_best_model_and_evaluate_binary.py"),
            "--csv", args.csv,
            "--model", best_model_finetune,
            "--outdir", best_results_dir,
            "--n_folds", str(args.n_splits),
            "--features_per_fold", features_per_fold_path
        ]

        try:
            subprocess.run(fine_tune_cmd, check=True)
            print(f"\n>>> Fine-tuning completado correctamente.")
            print(f">>> Resultados disponibles en: {best_results_dir}")
        except subprocess.CalledProcessError as e:
            print("\n[ERROR] El Script 3 falló durante la ejecución.")
            print(f"Comando ejecutado: {' '.join(fine_tune_cmd)}")
            print(f"Código de salida: {e.returncode}")
            raise

    print(f"\n>>> [FIN DEL RUN] Todos los procesos han finalizado exitosamente.")
    print(f">>> Directorio de resultados: {experiment_dir}")