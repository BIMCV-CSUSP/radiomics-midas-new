#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 3: fine-tuning, calibración, thresholding y explicabilidad
del mejor modelo, alineado con un pipeline SIN data leakage.

Características principales:
1) Split train/test por grupos
2) Selección final de variables SOLO en train
3) BayesSearchCV SOLO en train
4) Selección del threshold SOLO con predicciones OOF en train
5) Evaluación final UNA sola vez en test
6) SHAP sobre el modelo final entrenado

Nota:
- Este script ya no depende de variables_usadas.txt.
- Opcionalmente puede usar features_per_fold.csv para derivar una firma estable.
"""

import os
import argparse
import warnings
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(["science", "grid"])
mpl.rcParams["text.usetex"] = False
dpi = 300

import shap
import joblib

from scipy.special import expit
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, f1_score,
    accuracy_score, recall_score, precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    brier_score_loss, roc_curve
)

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


# ==============================================================================
# UTILIDADES GENERALES
# ==============================================================================

def save_plot_both_formats(fig_path_base, dpi=300, bbox_inches="tight"):
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
    if len(x) < 3 or x.nunique() < 2:
        return 1.0
    if len(x) > max_n:
        x = x.sample(max_n, random_state=random_state)
    try:
        _, p = shapiro(x)
    except Exception:
        p = 1.0
    return p


def clean_input_dataframe(df):
    """
    Limpieza consistente con el pipeline principal.
    """
    cols_to_drop = [
        "id_igtp", "patient_id", "study_id",
        "label", "mask_type", "SSA_type"
    ]

    if "label" not in df.columns:
        raise ValueError("El CSV debe contener una columna 'label'.")
    if "patient_id" not in df.columns:
        raise ValueError("El CSV debe contener una columna 'patient_id'.")

    y = df["label"].values
    groups = df["patient_id"].values

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if "diagnostics" in c], errors="ignore")

    # Nos quedamos solo con columnas numéricas
    X = X.select_dtypes(include=[np.number]).copy()

    return X, y, groups


# ==============================================================================
# SELECCIÓN FINAL DE VARIABLES (TRAIN-ONLY)
# ==============================================================================

def build_stable_feature_candidates(features_per_fold_path, min_frequency=0.50):
    """
    Construye una lista de variables candidatas "estables" a partir de features_per_fold.csv.
    Espera una columna 'Selected_Features' que contenga listas serializadas.
    """
    df_feat = pd.read_csv(features_per_fold_path)
    if "Selected_Features" not in df_feat.columns:
        raise ValueError("features_per_fold.csv debe contener la columna 'Selected_Features'.")

    all_lists = []
    for item in df_feat["Selected_Features"]:
        if pd.isna(item):
            continue
        if isinstance(item, str):
            try:
                parsed = eval(item)
            except Exception:
                parsed = []
        else:
            parsed = item
        if isinstance(parsed, (list, tuple)):
            all_lists.append(list(parsed))

    flat = [feat for lst in all_lists for feat in lst]
    counts = Counter(flat)

    n_folds = max(len(all_lists), 1)
    stable = [feat for feat, c in counts.items() if (c / n_folds) >= min_frequency]

    # Ordenamos por frecuencia descendente
    stable = sorted(stable, key=lambda f: counts[f], reverse=True)
    return stable, counts


def select_features_train_only(
    X_train,
    y_train,
    groups_train=None,
    corr_threshold=0.85,
    min_features=2,
    candidate_features=None,
    random_state=42
):
    """
    Selección final de variables SOLO en TRAIN.
    Pipeline:
      1) Screening univariante (p-value)
      2) Filtro de redundancia (Spearman)
      3) RFECV multivariante con Logistic L1 y CV por grupos
    """
    X_work = X_train.copy()

    if candidate_features is not None:
        candidate_features = [f for f in candidate_features if f in X_work.columns]
        if len(candidate_features) > 0:
            X_work = X_work[candidate_features].copy()

    # -------- 1) Ranking univariante (TRAIN ONLY) --------
    pvals = {}
    aucs = {}

    for col in X_work.columns:
        x_col = X_work[col]

        # Si la columna es constante o inválida, penalizarla
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
            fpr, tpr, _ = roc_curve(y_train, x_col, pos_label=1)
            auc_val = np.trapz(tpr, fpr)
            if auc_val < 0.5:
                fpr, tpr, _ = roc_curve(y_train, x_col, pos_label=0)
                auc_val = np.trapz(tpr, fpr)
        except Exception:
            auc_val = 0.5

        pvals[col] = pval
        aucs[col] = auc_val

    ranked_cols = sorted(pvals.keys(), key=lambda c: (pvals[c], -aucs[c]))
    X_ranked = X_work[ranked_cols]

    # -------- 2) Filtro de correlación --------
    corr = X_ranked.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X_clean = X_ranked.drop(columns=to_drop, errors="ignore")

    if X_clean.shape[1] < min_features:
        selected = ranked_cols[:min_features]
        return selected, {
            "ranked_cols": ranked_cols,
            "to_drop_corr": to_drop,
            "n_after_corr": X_clean.shape[1]
        }

    # -------- 3) RFECV multivariante con grupos --------
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

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

    selector = RFECV(
        estimator=base_model,
        step=1,
        cv=inner_cv,
        scoring="roc_auc",
        min_features_to_select=min_features,
        n_jobs=-1
    )

    if groups_train is None:
        # fallback si no hay grupos
        selector.fit(X_scaled, y_train)
    else:
        selector.fit(X_scaled, y_train, groups=groups_train)

    selected_features = X_clean.columns[selector.support_].tolist()

    if len(selected_features) < min_features:
        selected_features = ranked_cols[:min_features]

    info = {
        "ranked_cols": ranked_cols,
        "to_drop_corr": to_drop,
        "n_after_corr": X_clean.shape[1]
    }
    return selected_features, info


# ==============================================================================
# MODELOS Y ESPACIOS DE BÚSQUEDA
# ==============================================================================

def get_model_and_search_space(selected_model, random_state=42):
    """
    Devuelve pipeline y espacio de búsqueda Bayes para el modelo elegido.
    """
    if selected_model == "SVM":
        pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            SVC(random_state=random_state, probability=True, class_weight="balanced")
        )
        param_grid = {
            "svc__C": Real(1e-4, 1e3, prior="log-uniform"),
            "svc__kernel": Categorical(["linear", "rbf", "poly"]),
            "svc__gamma": Real(1e-4, 1e3, prior="log-uniform"),
            "svc__coef0": Real(0.0, 1.0)
        }

    elif selected_model == "LogisticRegression":
        pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            LogisticRegression(
                class_weight="balanced",
                random_state=random_state,
                solver="saga",
                max_iter=10000
            )
        )
        param_grid = {
            "logisticregression__C": Real(1e-4, 1e3, prior="log-uniform"),
            "logisticregression__penalty": Categorical(["l1", "l2", "elasticnet"]),
            "logisticregression__l1_ratio": Real(0.1, 0.9)
        }

    elif selected_model == "RandomForest":
        pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            RandomForestClassifier(
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=random_state
            )
        )
        param_grid = {
            "randomforestclassifier__n_estimators": Integer(100, 800),
            "randomforestclassifier__max_depth": Integer(2, 12),
            "randomforestclassifier__max_features": Categorical(["sqrt", "log2", None]),
            "randomforestclassifier__min_samples_split": Integer(2, 20)
        }

    elif selected_model == "NaiveBayes":
        pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            GaussianNB()
        )
        # Le damos al menos un parámetro para BayesSearchCV
        param_grid = {
            "gaussiannb__var_smoothing": Real(1e-12, 1e-6, prior="log-uniform")
        }

    elif selected_model == "KNN":
        pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            KNeighborsClassifier(n_jobs=-1)
        )
        param_grid = {
            "kneighborsclassifier__n_neighbors": Integer(2, 12),
            "kneighborsclassifier__weights": Categorical(["uniform", "distance"])
        }

    elif selected_model == "GradientBoosting":
        pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            GradientBoostingClassifier(random_state=random_state)
        )
        param_grid = {
            "gradientboostingclassifier__n_estimators": Integer(50, 600),
            "gradientboostingclassifier__learning_rate": Real(1e-4, 0.2, prior="log-uniform"),
            "gradientboostingclassifier__max_depth": Integer(1, 6),
            "gradientboostingclassifier__subsample": Real(0.5, 1.0),
            "gradientboostingclassifier__max_features": Categorical(["sqrt", "log2", None])
        }

    else:
        raise ValueError(f"Modelo no reconocido: {selected_model}")

    return pipe, param_grid


# ==============================================================================
# SCORES, CALIBRACIÓN Y THRESHOLD
# ==============================================================================

def get_raw_scores(fitted_estimator, X):
    """
    Devuelve un score continuo:
    - predict_proba[:,1] si existe
    - decision_function si no
    - error si no existe ninguno
    """
    if hasattr(fitted_estimator, "predict_proba"):
        return fitted_estimator.predict_proba(X)[:, 1]
    elif hasattr(fitted_estimator, "decision_function"):
        return fitted_estimator.decision_function(X)
    else:
        raise ValueError("El estimador no implementa ni predict_proba ni decision_function.")


def get_probability_like(fitted_estimator, X):
    """
    Probabilidad aproximada para curvas de calibración / Brier:
    - predict_proba[:,1] si existe
    - sigmoid(decision_function) si no (solo como proxy pre-calibración)
    """
    if hasattr(fitted_estimator, "predict_proba"):
        return fitted_estimator.predict_proba(X)[:, 1]
    elif hasattr(fitted_estimator, "decision_function"):
        return expit(fitted_estimator.decision_function(X))
    else:
        raise ValueError("No se puede obtener una probabilidad aproximada del estimador.")


def get_oof_scores(estimator, X, y, groups, cv):
    """
    Obtiene scores out-of-fold (OOF) sobre TRAIN para:
    - calibración tipo Platt (train-only)
    - selección de threshold (train-only)
    """
    oof_scores = np.full(shape=len(y), fill_value=np.nan, dtype=float)

    for tr_idx, va_idx in cv.split(X, y, groups=groups):
        est = clone(estimator)
        est.fit(X.iloc[tr_idx], y[tr_idx])

        scores = get_raw_scores(est, X.iloc[va_idx])
        oof_scores[va_idx] = scores

    return oof_scores


def fit_platt_scaler(oof_scores, y_true, random_state=42):
    """
    Ajuste de Platt scaling sobre scores OOF del train.
    """
    scores = np.asarray(oof_scores).reshape(-1, 1)
    y_true = np.asarray(y_true)

    lr = LogisticRegression(
        solver="lbfgs",
        random_state=random_state,
        max_iter=1000
    )
    lr.fit(scores, y_true)
    return lr


def apply_platt_scaler(platt_model, raw_scores):
    scores = np.asarray(raw_scores).reshape(-1, 1)
    return platt_model.predict_proba(scores)[:, 1]


def calibration_error(y_true, y_prob, n_bins=10, norm="l1"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if norm not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    err_total = 0.0
    N = len(y_true)

    for i in range(n_bins):
        idx = (bin_ids == i)
        if idx.any():
            prob_avg = y_prob[idx].mean()
            acc_avg = y_true[idx].mean()
            err_bin = abs(prob_avg - acc_avg) if norm == "l1" else (prob_avg - acc_avg) ** 2
            err_total += err_bin * idx.sum() / N

    return err_total


def find_best_threshold_from_train_probs(y_train, prob_train, metric="f1"):
    """
    Selecciona threshold SOLO usando train (OOF probabilities).
    """
    thresholds = np.linspace(0.1, 0.9, 81)  # más fino que 0.1,0.2,...
    best_threshold = 0.5
    best_score = -np.inf
    rows = []

    for thr in thresholds:
        y_pred_thr = (prob_train >= thr).astype(int)

        if metric == "f1":
            score = f1_score(y_train, y_pred_thr)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_train, y_pred_thr)
        else:
            raise ValueError("metric must be 'f1' or 'balanced_accuracy'.")

        rows.append({"threshold": thr, "score": score})

        if score > best_score:
            best_score = score
            best_threshold = thr

    return best_threshold, best_score, pd.DataFrame(rows)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Métricas robustas para binaria.
    """
    out = {}
    out["AUC"] = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    out["MCC"] = matthews_corrcoef(y_true, y_pred)
    out["Kappa"] = cohen_kappa_score(y_true, y_pred)
    out["F1"] = f1_score(y_true, y_pred)
    out["Accuracy"] = accuracy_score(y_true, y_pred)
    out["Sensitivity"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    out["Specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    out["PPV"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    out["NPV"] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    out["Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    return out


# ==============================================================================
# SHAP
# ==============================================================================

def perform_shap_analysis(X_data, y_data, model_clf, preprocessor, shap_dir, report_path, dataset_name="dataset"):
    """
    SHAP sobre datos ya restringidos a la firma final.
    """
    print(f"\n[SHAP] Analizando {dataset_name}...")
    os.makedirs(shap_dir, exist_ok=True)

    try:
        # Transformación manual conservando nombres
        scaler = preprocessor.steps[0][1]
        X_scaled = pd.DataFrame(
            scaler.transform(X_data),
            index=X_data.index,
            columns=X_data.columns
        )

        vt = preprocessor.steps[1][1]
        mask = vt.get_support()
        selected_features = X_data.columns[mask]
        X_transformed = pd.DataFrame(
            vt.transform(X_scaled.values),
            index=X_data.index,
            columns=selected_features
        )

        shap_cache = os.path.join(shap_dir, "shap_values.pkl")

        if os.path.exists(shap_cache):
            shap_values = joblib.load(shap_cache)
        else:
            if isinstance(model_clf, (RandomForestClassifier, GradientBoostingClassifier)):
                explainer = shap.TreeExplainer(model_clf)
            elif isinstance(model_clf, LogisticRegression):
                try:
                    explainer = shap.LinearExplainer(model_clf, X_transformed)
                except Exception:
                    background = shap.kmeans(X_transformed, 50)
                    explainer = shap.KernelExplainer(model_clf.predict_proba, background)
            else:
                background = shap.kmeans(X_transformed, 50)
                explainer = shap.KernelExplainer(model_clf.predict_proba, background)

            shap_values = explainer(X_transformed)
            joblib.dump(shap_values, shap_cache)

        # Binaria: quedarnos con la clase positiva
        if hasattr(shap_values, "values") and shap_values.values.ndim > 2:
            shap_values = shap_values[:, :, 1]

        # -------------------------------
        # Test estadístico por feature
        # -------------------------------
        shap_matrix = pd.DataFrame(
            shap_values.values,
            index=X_transformed.index,
            columns=X_transformed.columns
        )

        features_test = []
        pvalues_raw = []

        for feat in shap_matrix.columns:
            shap_class0 = shap_matrix.loc[y_data == 0, feat]
            shap_class1 = shap_matrix.loc[y_data == 1, feat]

            try:
                _, pval = mannwhitneyu(shap_class0, shap_class1, alternative="two-sided")
            except Exception:
                pval = 1.0

            features_test.append(feat)
            pvalues_raw.append(pval)

        alpha = 0.05
        reject, pvals_corr, _, _ = multipletests(pvalues_raw, alpha=alpha, method="holm")

        test_txt_path = os.path.join(shap_dir, "shap_statistical_test.txt")
        with open(test_txt_path, "w", encoding="utf-8") as f_out:
            f_out.write("=================================\n")
            f_out.write(f"MANN-WHITNEY U TEST (SHAP) for {dataset_name}\n")
            f_out.write("Correction: Holm\n")
            f_out.write(f"alpha = {alpha}\n")
            f_out.write("=================================\n\n")
            for feat, p_raw, p_corr, rej in zip(features_test, pvalues_raw, pvals_corr, reject):
                tag = "SIGNIFICANT" if rej else "ns"
                f_out.write(
                    f"{feat}: raw={p_raw:.4e}, corrected={p_corr:.4e} -> {tag}\n"
                )

        # -------------------------------
        # Heatmap
        # -------------------------------
        idx_class0 = np.where(y_data == 0)[0]
        idx_class1 = np.where(y_data == 1)[0]
        idx_order = np.concatenate([idx_class0, idx_class1])

        shap.plots.heatmap(shap_values, show=False, instance_order=idx_order)
        fig = plt.gcf()
        ax = plt.gca()

        split_position = len(idx_class0)
        ax.axvline(split_position - 0.5, color="black", linewidth=1, zorder=10)

        n_total = len(idx_order)
        mid_class0 = (split_position / 2) / n_total
        mid_class1 = (split_position + len(idx_class1) / 2) / n_total

        ax.text(mid_class0, 1.01, "Class 0", ha="center", va="bottom", transform=ax.transAxes)
        ax.text(mid_class1, 1.01, "Class 1", ha="center", va="bottom", transform=ax.transAxes)

        fig.set_size_inches(10, 6)
        plt.tight_layout()
        save_plot_both_formats(os.path.join(shap_dir, "shap_heatmap"), dpi=dpi)
        plt.close()

        # -------------------------------
        # Beeswarm
        # -------------------------------
        shap.plots.beeswarm(shap_values, max_display=16, show=False)
        fig = plt.gcf()
        fig.set_size_inches(14, 8)
        plt.subplots_adjust(left=0.4, right=0.95)
        plt.tight_layout()
        save_plot_both_formats(os.path.join(shap_dir, "shap_beeswarm"), dpi=dpi)
        plt.close()

        # -------------------------------
        # Scatter top features
        # -------------------------------
        scatter_dir = os.path.join(shap_dir, "scatter_plots")
        os.makedirs(scatter_dir, exist_ok=True)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-15:]
        top_idx = top_idx[np.argsort(mean_abs_shap[top_idx])[::-1]]
        top_features_shap = X_transformed.columns[top_idx]

        for i, feature in enumerate(top_features_shap, start=1):
            shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            plt.tight_layout()
            safe_name = str(feature).replace("/", "_")
            save_plot_both_formats(os.path.join(scatter_dir, f"{i:02d}_{safe_name}"), dpi=dpi)
            plt.close()

        return True, list(selected_features), shap_values, list(top_features_shap)

    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"\n=== SHAP Analysis ({dataset_name}) ===\n")
            f_out.write(f"Error: {repr(e)}\n")
        print(f"[SHAP] Error en {dataset_name}: {e}")
        return False, None, None, None


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning, calibración y explicabilidad del mejor modelo (versión corregida)."
    )
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV con features.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["SVM", "LogisticRegression", "RandomForest", "NaiveBayes", "KNN", "GradientBoosting"],
        help="Modelo a optimizar."
    )
    parser.add_argument("--outdir", type=str, required=True, help="Directorio de salida.")
    parser.add_argument("--n_folds", type=int, default=5, help="Número de folds CV para tuning.")
    parser.add_argument("--test_size", type=float, default=0.20, help="Tamaño del hold-out test.")
    parser.add_argument("--random_state", type=int, default=42, help="Semilla global.")
    parser.add_argument(
        "--features_per_fold",
        type=str,
        default=None,
        help="CSV opcional con variables por fold (para derivar firma estable)."
    )
    parser.add_argument(
        "--stable_feature_min_freq",
        type=float,
        default=0.50,
        help="Frecuencia mínima de selección para considerar una variable estable."
    )
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="f1",
        choices=["f1", "balanced_accuracy"],
        help="Métrica usada para elegir el threshold SOLO en train."
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    calibration_dir = os.path.join(args.outdir, "calibration")
    explicability_dir = os.path.join(args.outdir, "explicability")
    train_shap_dir = os.path.join(explicability_dir, "train", "SHAP")
    test_shap_dir = os.path.join(explicability_dir, "test", "SHAP")

    os.makedirs(calibration_dir, exist_ok=True)
    os.makedirs(train_shap_dir, exist_ok=True)
    os.makedirs(test_shap_dir, exist_ok=True)

    report_path = os.path.join(args.outdir, "report.txt")

    print("\n=== INICIO SCRIPT 3 CORREGIDO ===")
    print(f"Modelo: {args.model}")
    print(f"CSV: {args.csv}")
    print(f"Outdir: {args.outdir}")

    # ----------------------------------------------------------------------
    # 1) CARGA Y LIMPIEZA
    # ----------------------------------------------------------------------
    df = pd.read_csv(args.csv)
    X, y, groups = clean_input_dataframe(df)

    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write("=== FINAL MODEL SCRIPT (CORREGIDO) ===\n\n")
        f_out.write(f"Model: {args.model}\n")
        f_out.write(f"CSV: {args.csv}\n")
        f_out.write(f"Samples: {X.shape[0]}\n")
        f_out.write(f"Initial features: {X.shape[1]}\n\n")

    # ----------------------------------------------------------------------
    # 2) HOLD-OUT TRAIN/TEST POR GRUPOS
    # ----------------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_full = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train_full = y[train_idx]
    y_test = y[test_idx]
    groups_train_full = groups[train_idx]
    groups_test = groups[test_idx]

    print(f"Train shape: {X_train_full.shape} | Test shape: {X_test.shape}")

    # ----------------------------------------------------------------------
    # 3) OPCIONAL: DERIVAR CANDIDATAS ESTABLES DESDE features_per_fold.csv
    # ----------------------------------------------------------------------
    candidate_features = None
    if args.features_per_fold is not None and os.path.exists(args.features_per_fold):
        stable_features, counts = build_stable_feature_candidates(
            args.features_per_fold,
            min_frequency=args.stable_feature_min_freq
        )
        candidate_features = [f for f in stable_features if f in X_train_full.columns]

        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write("=== Stable feature candidates ===\n")
            f_out.write(f"features_per_fold: {args.features_per_fold}\n")
            f_out.write(f"min_frequency: {args.stable_feature_min_freq}\n")
            f_out.write(f"stable candidates found: {len(candidate_features)}\n")
            if len(candidate_features) > 0:
                f_out.write("Top stable candidates:\n")
                for feat in candidate_features[:50]:
                    f_out.write(f"  - {feat} (count={counts[feat]})\n")
            f_out.write("\n")

    # ----------------------------------------------------------------------
    # 4) SELECCIÓN FINAL DE VARIABLES SOLO EN TRAIN
    # ----------------------------------------------------------------------
    print("\n[Feature selection] Selección final SOLO en train...")
    final_features, fs_info = select_features_train_only(
        X_train=X_train_full,
        y_train=y_train_full,
        groups_train=groups_train_full,
        corr_threshold=0.85,
        min_features=2,
        candidate_features=candidate_features,
        random_state=args.random_state
    )

    X_train_final = X_train_full[final_features].copy()
    X_test_final = X_test[final_features].copy()

    final_features_txt = os.path.join(args.outdir, "final_selected_features.txt")
    with open(final_features_txt, "w", encoding="utf-8") as f:
        for feat in final_features:
            f.write(f"{feat}\n")

    pd.DataFrame({"Feature": final_features}).to_csv(
        os.path.join(args.outdir, "final_selected_features.csv"),
        index=False
    )

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Final feature selection (TRAIN ONLY) ===\n")
        f_out.write(f"Final number of features: {len(final_features)}\n")
        f_out.write("Selected features:\n")
        for feat in final_features:
            f_out.write(f"  - {feat}\n")
        f_out.write("\n")

    print(f"[Feature selection] Variables finales: {len(final_features)}")

    # ----------------------------------------------------------------------
    # 5) PIPELINE + BAYESSEARCH SOLO EN TRAIN
    # ----------------------------------------------------------------------
    print("\n[BayesSearchCV] Iniciando tuning...")
    pipe, param_grid = get_model_and_search_space(args.model, random_state=args.random_state)

    inner_cv = StratifiedGroupKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.random_state
    )

    score_group = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy"
    }
    score_refit = "roc_auc"

    estimator_path = os.path.join(args.outdir, "best_estimator.pkl")
    search_path = os.path.join(args.outdir, "search_results.pkl")

    if os.path.exists(estimator_path) and os.path.exists(search_path):
        print(f">>> Modelo pre-entrenado encontrado en {args.outdir}. Cargando...")
        best_estimator = joblib.load(estimator_path)
        search = joblib.load(search_path)
    else:
        search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_grid,
            scoring=score_group,
            refit=score_refit,
            cv=inner_cv,
            n_jobs=-1,
            random_state=args.random_state,
            return_train_score=False
        )
        search.fit(X_train_final, y_train_full, groups=groups_train_full)

        best_estimator = search.best_estimator_
        joblib.dump(best_estimator, estimator_path)
        joblib.dump(search, search_path)

    print(f"[BayesSearchCV] Mejores parámetros: {search.best_params_}")

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== BayesSearchCV results ===\n")
        f_out.write(f"Best params: {search.best_params_}\n")
        idx_best = search.best_index_
        for key in score_group:
            mean_test = search.cv_results_[f"mean_test_{key}"][idx_best]
            std_test = search.cv_results_[f"std_test_{key}"][idx_best]
            f_out.write(f"CV {key}: {mean_test:.3f} +/- {std_test:.3f}\n")
        f_out.write("\n")

    # ----------------------------------------------------------------------
    # 6) CALIBRACIÓN Y THRESHOLD SOLO EN TRAIN (OOF)
    # ----------------------------------------------------------------------
    print("\n[Calibration] Obteniendo scores OOF en train...")
    oof_scores = get_oof_scores(
        estimator=best_estimator,
        X=X_train_final,
        y=y_train_full,
        groups=groups_train_full,
        cv=inner_cv
    )

    # Platt scaling SOLO en train
    platt_model = fit_platt_scaler(oof_scores, y_train_full, random_state=args.random_state)
    joblib.dump(platt_model, os.path.join(args.outdir, "platt_scaler.pkl"))

    oof_probs_cal = apply_platt_scaler(platt_model, oof_scores)

    # Threshold SOLO con train OOF calibrado
    best_thresh, best_thr_score, df_thr = find_best_threshold_from_train_probs(
        y_train=y_train_full,
        prob_train=oof_probs_cal,
        metric=args.threshold_metric
    )

    df_thr.to_csv(os.path.join(args.outdir, "threshold_search_train.csv"), index=False)

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Threshold selection (TRAIN OOF ONLY) ===\n")
        f_out.write(f"Metric used: {args.threshold_metric}\n")
        f_out.write(f"Best threshold: {best_thresh:.3f}\n")
        f_out.write(f"Best train score: {best_thr_score:.3f}\n\n")

    # ----------------------------------------------------------------------
    # 7) REFIT FINAL EN TODO TRAIN
    # ----------------------------------------------------------------------
    print("\n[Refit] Ajustando estimador final en TODO el train...")
    best_estimator.fit(X_train_final, y_train_full)
    raw_train = get_raw_scores(best_estimator, X_train_final)
    raw_test = get_raw_scores(best_estimator, X_test_final)

    prob_train_pre = get_probability_like(best_estimator, X_train_final)
    prob_test_pre = get_probability_like(best_estimator, X_test_final)

    prob_train_post = apply_platt_scaler(platt_model, raw_train)
    prob_test_post = apply_platt_scaler(platt_model, raw_test)

    y_pred_test_default = (prob_test_post >= 0.5).astype(int)
    y_pred_test_best = (prob_test_post >= best_thresh).astype(int)

    # ----------------------------------------------------------------------
    # 8) EVALUACIÓN FINAL EN TEST
    # ----------------------------------------------------------------------
    print("\n[Test] Evaluación final en test...")
    metrics_default = compute_metrics(y_test, y_pred_test_default, prob_test_post)
    metrics_best = compute_metrics(y_test, y_pred_test_best, prob_test_post)

    report_default = classification_report(y_test, y_pred_test_default)
    report_best = classification_report(y_test, y_pred_test_best)

    # Guardar predicciones
    df_test_preds = pd.DataFrame({
        "y_true": y_test,
        "prob_pre": prob_test_pre,
        "prob_post": prob_test_post,
        "pred_default_0_5": y_pred_test_default,
        "pred_best_threshold": y_pred_test_best
    })
    df_test_preds.to_csv(os.path.join(args.outdir, "test_predictions.csv"), index=False)

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Final test evaluation ===\n")
        f_out.write("(Probabilities calibrated with Platt scaling fit on TRAIN OOF scores)\n\n")

        f_out.write("[Threshold = 0.50]\n")
        for k, v in metrics_default.items():
            f_out.write(f"{k}: {v:.3f}\n")
        f_out.write("\nClassification report:\n")
        f_out.write(report_default)
        f_out.write("\n\n")

        f_out.write(f"[Threshold = best train threshold = {best_thresh:.3f}]\n")
        for k, v in metrics_best.items():
            f_out.write(f"{k}: {v:.3f}\n")
        f_out.write("\nClassification report:\n")
        f_out.write(report_best)
        f_out.write("\n\n")

    # ----------------------------------------------------------------------
    # 9) MATRICES DE CONFUSIÓN
    # ----------------------------------------------------------------------
    cm_default = confusion_matrix(y_test, y_pred_test_default)
    cm_best = confusion_matrix(y_test, y_pred_test_best)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_default)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test confusion matrix (threshold = 0.50)")
    plt.tight_layout()
    save_plot_both_formats(os.path.join(args.outdir, "confusion_matrix_test_default"), dpi=dpi)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_best)
    disp.plot(ax=ax, cmap="cividis", colorbar=False)
    ax.set_title(f"Test confusion matrix (threshold = {best_thresh:.2f})")
    plt.tight_layout()
    save_plot_both_formats(os.path.join(calibration_dir, "confusion_matrix_test_best_threshold"), dpi=dpi)
    plt.close()

    # ----------------------------------------------------------------------
    # 10) CURVAS DE CALIBRACIÓN + MÉTRICAS DE CALIBRACIÓN
    # ----------------------------------------------------------------------
    print("\n[Calibration] Curvas y métricas...")

    # Curva pre
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_predictions(
        y_true=y_test,
        y_prob=prob_test_pre,
        n_bins=10,
        name=f"{args.model}_pre",
        ax=ax
    )
    ax.set_title("Calibration curve (pre)")
    plt.tight_layout()
    save_plot_both_formats(os.path.join(calibration_dir, "calibration_pre"), dpi=dpi)
    plt.close()

    # Curva post
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_predictions(
        y_true=y_test,
        y_prob=prob_test_post,
        n_bins=10,
        name=f"{args.model}_post",
        ax=ax
    )
    ax.set_title("Calibration curve (post)")
    plt.tight_layout()
    save_plot_both_formats(os.path.join(calibration_dir, "calibration_post"), dpi=dpi)
    plt.close()

    ece_pre = calibration_error(y_test, prob_test_pre, n_bins=10, norm="l1")
    ece_post = calibration_error(y_test, prob_test_post, n_bins=10, norm="l1")
    brier_pre = brier_score_loss(y_test, prob_test_pre)
    brier_post = brier_score_loss(y_test, prob_test_post)

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Calibration metrics (TEST) ===\n")
        f_out.write(f"ECE pre:  {ece_pre:.3f}\n")
        f_out.write(f"ECE post: {ece_post:.3f}\n")
        f_out.write(f"Brier pre:  {brier_pre:.3f}\n")
        f_out.write(f"Brier post: {brier_post:.3f}\n\n")

    # ----------------------------------------------------------------------
    # 11) SHAP SOBRE MODELO FINAL
    # ----------------------------------------------------------------------
    print("\n[SHAP] Interpretabilidad del modelo final...")

    preprocessor = deepcopy(best_estimator)
    preprocessor.steps.pop(-1)
    model_clf = best_estimator.steps[-1][1]

    train_success, selected_features_shap_train, train_shap_values, train_top_feats = perform_shap_analysis(
        X_data=X_train_final,
        y_data=y_train_full,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=train_shap_dir,
        report_path=report_path,
        dataset_name="train"
    )

    test_success, selected_features_shap_test, test_shap_values, test_top_feats = perform_shap_analysis(
        X_data=X_test_final,
        y_data=y_test,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=test_shap_dir,
        report_path=report_path,
        dataset_name="test"
    )

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== SHAP summary ===\n")
        f_out.write(f"Train SHAP success: {train_success}\n")
        f_out.write(f"Test SHAP success: {test_success}\n")
        if train_top_feats is not None:
            f_out.write(f"Top train SHAP features: {train_top_feats}\n")
        if test_top_feats is not None:
            f_out.write(f"Top test SHAP features: {test_top_feats}\n")
        f_out.write("\n")

    print("\n=== FIN SCRIPT 3 CORREGIDO ===")
    print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()