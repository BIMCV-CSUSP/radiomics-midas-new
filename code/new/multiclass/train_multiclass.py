#### LIBRARIES
import argparse
import pandas as pd
import numpy as np
import os

from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn import metrics

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, confusion_matrix)
from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
from sklearn.preprocessing import label_binarize
mpl.use('Agg')
import scienceplots

plt.style.use(['science', 'grid'])
dpi = 300
from scipy.stats import kruskal, f_oneway
plt.rcParams["text.usetex"] = False
import ast
import seaborn as sns
import subprocess



##### FUNCTIONS
def get_models(random_state=42):
    """
    Define the pipelines for each classifier, including standard preprocessing.
    
    Args:
        random_state (int): Seed for reproducibility.
    
    Returns:
        list: List of tuples (model_name, scikit_learn_pipeline)
    """

    # Pipeline for Support Vector Machine
    pipe_svc = make_pipeline(
        StandardScaler(),  # Feature scaling
        VarianceThreshold(),  # Remove zero-variance features
        SVC(random_state=random_state, class_weight="balanced", probability=True)
    )
    
    # Pipeline for Logistic Regression
    pipe_lr = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        LogisticRegression(
            penalty='elasticnet',       # Combined L1 & L2 regularization
            l1_ratio=0.5,               # Balance between L1 and L2
            class_weight="balanced",
            random_state=random_state,
            solver='saga',              # Optimizer for elasticnet
            max_iter=10000              # Max iterations
        )
    )
    
    # Pipeline for Random Forest
    pipe_rf = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", random_state=random_state)
    )
    
    # Pipeline for Gaussian Naive Bayes
    pipe_nb = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        GaussianNB()  # No extra parameters needed
    )
    
    # Pipeline for K-Nearest Neighbors
    pipe_knn = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        KNeighborsClassifier(n_jobs=-1)
    )
    
    # Pipeline for Gradient Boosting
    pipe_gb = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        GradientBoostingClassifier(random_state=random_state)
    )

    # List of all models
    models = [
        ("SVM", pipe_svc),
        ("Logistic Regression", pipe_lr),
        ("Random Forest", pipe_rf),
        ("Naive Bayes", pipe_nb),
        ("KNN", pipe_knn),
        ("Gradient Boosting", pipe_gb),
    ]
    return models


def evaluate_model_multiclass(model, X, y, groups, n_splits=5, n_repeats=1, base_random_state=42):
    """
    Perform repeated stratified group cross-validation for multiclass classification.
    
    Args:
        model: Model to evaluate (scikit-learn pipeline)
        X (pd.DataFrame): Features
        y (np.array): Multiclass labels
        groups (np.array): Group identifiers (patients) for CV
        n_splits (int): Number of folds per repetition
        n_repeats (int): Number of repetitions
        base_random_state (int): Seed for reproducibility
    
    Returns:
        tuple: (fold_results, pred_vals)
            - fold_results: List of dicts with metrics per fold
            - pred_vals: Dict with prediction data for each fold
    """
    fold_results = []
    folds_data = []
    global_fold_index = 0
    classes = np.unique(y)
    for rep in range(n_repeats):
        current_random_state = base_random_state + rep
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=current_random_state
        )
        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            global_fold_index += 1
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)

            # Probabilities or scores
            if hasattr(model, "predict_proba"):
                y_train_prob = model.predict_proba(X_train)
            elif hasattr(model, "decision_function"):
                y_train_prob = model.decision_function(X_train)
            else:
                y_train_prob = None

            # AUC multiclase en entrenamiento
            try:
                y_train_bin = label_binarize(y_train, classes=classes)
                if y_train_prob is not None and len(np.unique(y_train)) > 1:
                    train_auc = roc_auc_score(y_train_bin, y_train_prob, multi_class="ovr", average="macro")
                else:
                    train_auc = np.nan
            except:
                train_auc = np.nan

            train_f1_macro = f1_score(y_train, y_train_pred, average="macro")

            # Validación
            y_val_pred = model.predict(X_val)
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(X_val)
            elif hasattr(model, "decision_function"):
                y_val_prob = model.decision_function(X_val)
            else:
                y_val_prob = None

            # AUC multiclase en validación
            try:
                y_val_bin = label_binarize(y_val, classes=classes)
                if y_val_prob is not None and len(np.unique(y_val)) > 1:
                    val_auc = roc_auc_score(y_val_bin, y_val_prob, multi_class="ovr", average="macro")
                else:
                    val_auc = np.nan
            except:
                val_auc = np.nan

            val_mcc = matthews_corrcoef(y_val, y_val_pred)
            val_kappa = cohen_kappa_score(y_val, y_val_pred)
            val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)

            # Métricas por clase
            per_class_precision = precision_score(y_val, y_val_pred, average=None, labels=classes)
            per_class_recall = recall_score(y_val, y_val_pred, average=None, labels=classes)
            per_class_f1 = f1_score(y_val, y_val_pred, average=None, labels=classes)

            # Matriz de confusión y exactitud por clase
            cm = confusion_matrix(y_val, y_val_pred, labels=classes)
            per_class_accuracy = []
            for i in range(len(cm)):
                row_sum = np.sum(cm[i, :])
                if row_sum > 0:
                    per_class_accuracy.append(cm[i, i] / row_sum)
                else:
                    per_class_accuracy.append(np.nan)

            fold_metrics = {
                "Fold": global_fold_index,
                "Repeat": rep + 1,
                "train_auc": train_auc,
                "train_f1_macro": train_f1_macro,
                "val_auc": val_auc,
                "val_mcc": val_mcc,
                "val_kappa": val_kappa,
                "val_f1_macro": val_f1_macro,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": val_balanced_accuracy,
                "per_class_precision": per_class_precision.tolist(),
                "per_class_recall": per_class_recall.tolist(),
                "per_class_f1": per_class_f1.tolist(),
                "per_class_accuracy": per_class_accuracy
            }
            fold_results.append(fold_metrics)

            folds_data.append({
                "fold_index": global_fold_index,
                "Repeat": rep + 1,
                "y_val": y_val,
                "y_val_pred": y_val_pred,
                "y_val_prob": y_val_prob
            })

    pred_vals = {
        "folds": folds_data
    }
    return fold_results, pred_vals

##### MAIN CODE
from datetime import datetime
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiclass Radiomics Pipeline")
    
    # Argumentos configurables
    parser.add_argument("--csv", type=str, default="/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/data/multiclass/features_t2w_MPfirrmann.csv")
    parser.add_argument("--results_base", type=str, default="/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/data/multiclass")
    parser.add_argument("--resume_run", type=str, default=None, help="Name of the previous run folder")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=10)    
    args = parser.parse_args()

    # --- 1. Gestión Dinámica de Directorios ---
    if args.resume_run:
        experiment_dir = os.path.join(args.results_base, "runs", args.resume_run)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(args.results_base, "runs", f"run_multi_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
    
    print(f">>> Work directory: {experiment_dir}")

    # --- 2. Carga y Limpieza (Automatizada) ---
    df = pd.read_csv(args.csv)
    y = df["label"]
    groups = df["patient_id"]
    
    # Lista de columnas a eliminar (incluye todas las de diagnóstico automáticamente)
    cols_to_drop = ['id_igtp','patient_id', 'study_id', 'label', 'mask_type'] + [c for c in df.columns if 'diagnostics' in c]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    print(f">> Datos cargados. Formato: {X.shape}")

# --- Feature selection with persistence and redundancy handling ---
    selected_features = X.columns
    fs_dir = os.path.join(experiment_dir, "feature_selection")
    df_path_1 = os.path.join(fs_dir, "train_pvals_multiclass.csv")

    # 1. Check if statistical analysis already exists (Persistence)
    if os.path.exists(df_path_1):
        print(f">> Loading existing feature selection from: {df_path_1}")
        train_auc_pvals_df = pd.read_csv(df_path_1, index_col=0)
        # Select N best features based on p-value and AUC (max 1 feature per 15 samples)
        num_features_model = round(X.shape[0] / 2)
        selected_features = train_auc_pvals_df.index[:num_features_model].tolist()
        X = X[selected_features]
    else:
        print(">> No previous analysis found. Starting feature selection (Statistical + Discriminative)...")
        os.makedirs(fs_dir, exist_ok=True)
        images_dir = os.path.join(fs_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        feature_names, pvalue_list, auc_list = [], [], []
        
        # Evaluate each feature individually
        for column in X.columns:
            # Shapiro-Wilk test for normality
            stat, p = shapiro(X[column])
            groups_ = [X[column][y == clase] for clase in np.unique(y)]
            
            # Statistical test: ANOVA (normal) or Kruskal-Wallis (non-parametric)
            if p > 0.05:
                _, pval = f_oneway(*groups_)
            else:
                _, pval = kruskal(*groups_)
            
            # Discriminative capacity (Mean One-vs-Rest AUC)
            # Binarize labels for AUC multiclass calculation
            y_bin = label_binarize(y, classes=np.unique(y))
            auc_val = roc_auc_score(y_bin, X[column].values.reshape(-1, 1), multi_class='ovr', average='macro')
            
            feature_names.append(column)
            pvalue_list.append(pval)
            auc_list.append(auc_val)

        # Create DataFrame summarizing statistics
        train_auc_pvals_df = pd.DataFrame(
            list(zip(pvalue_list, auc_list)),
            index=feature_names,
            columns=['p-value', 'Mean_AUC']
        ).sort_values(by=['p-value', 'Mean_AUC'], ascending=[True, False])

        df_path_raw = os.path.join(fs_dir, "ranking_puro_sin_correlacion.csv")
        train_auc_pvals_df.to_csv(df_path_raw)
        print(f"  --> Guardado ranking inicial (sin filtro): {df_path_raw}")

        # --- FILTRO DE REDUNDANCIA (CORRELACIÓN) ---
        print("  --> Eliminando características redundantes (Spearman > 0.95)...")
        X_sorted = X[train_auc_pvals_df.index]
        corr_matrix = X_sorted.corr(method='spearman').abs()

        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.95:
                    colname = corr_matrix.columns[i]
                    to_drop.add(colname)

        # Aplicar limpieza
        train_df_filtered = train_auc_pvals_df.drop(index=list(to_drop))
        print(f"  --> {len(to_drop)} variables eliminadas por alta correlación.")


        # Select Top N: limit max 1 feature per 15 samples
        num_features_model = max(1, round(X.shape[0] / 2))
        selected_features = train_df_filtered.index[:num_features_model].tolist()
        
        # Save statistical report and filter X
        X = X[selected_features]
        train_df_filtered.to_csv(df_path_1)
        print(f"  --> Selected {len(selected_features)} relevant features. Saved report to: {df_path_1}")

        # --- Generate plots for TOP 20 features ---
        for rank, feature_name in enumerate(train_df_filtered.index[:20], start=1):
            safe_name = feature_name.replace("/", "_")
            path = os.path.join(images_dir, f"{rank}_{safe_name}")
            os.makedirs(path, exist_ok=True)
            
            # Violin plot to visualize distribution by class
            plt.figure(figsize=(8, 6))
            sns.violinplot(x=y, y=df[feature_name], palette="muted")
            plt.title(f"{feature_name} (AUC={train_df_filtered.loc[feature_name, 'Mean_AUC']:.2f})")
            plt.savefig(os.path.join(path, "violin.png"), dpi=dpi)
            plt.close()

    
# --- Model training & evaluation (Automated) ---
    resultados_filepath = os.path.join(experiment_dir, "resultados_discoslumbar.csv")
    preds_filepath = os.path.join(experiment_dir, "preds_discoslumbar.csv")
    variables_txt_path = os.path.join(experiment_dir, "variables_used.txt")

    # Persist the list of selected variables for the Fine-tuning script
    with open(variables_txt_path, "w") as f:
        for feat in selected_features:
            f.write(str(feat) + "\n")

    if os.path.exists(resultados_filepath) and os.path.exists(preds_filepath):
        print(f"\n>>> Results found in {experiment_dir}. Skipping training phase...")
        df_resultados = pd.read_csv(resultados_filepath)
        # Note: Load predictions with converters for list columns if needed later
    else:
        print("\n>>> No previous results found. Starting model training...")
        models = get_models(random_state=42)
        all_results, all_preds = [], []

        for model_name, model in models:
            print(f"Evaluating {model_name}...")
            fold_metrics_list, pred_vals = evaluate_model_multiclass(
                model, X, y, groups,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                base_random_state=42
            )

            # Store metrics
            for m in fold_metrics_list:
                m["Classifier"] = model_name
                all_results.append(m)

            # Store predictions
            for p in pred_vals["folds"]:
                p["Classifier"] = model_name
                all_preds.append(p)

        # Save metrics
        df_resultados = pd.DataFrame(all_results)
        df_resultados.sort_values(by=["Classifier", "Fold"], inplace=True)
        df_resultados.to_csv(resultados_filepath, index=False)
        print(f"Results saved at: {resultados_filepath}")

        # Save predictions
        records_for_csv = []
        for p in all_preds:
            records_for_csv.append({
                "Classifier": p["Classifier"],
                "Fold": p["fold_index"],
                "Repeat": p["Repeat"],
                "y_val": p["y_val"].tolist(),
                "y_pred": p["y_val_pred"].tolist(),
                "y_prob": p["y_val_prob"].tolist() if p["y_val_prob"] is not None else []
            })
        df_preds = pd.DataFrame(records_for_csv)
        df_preds.to_csv(preds_filepath, index=False)
        print(f"Predictions saved at: {preds_filepath}")

    # # --- Generate multiclass ROC curves (One-vs-Rest): optimal and median folds ---
    
    # --- ROC curves generation (Multiclass One-vs-Rest) ---
    roc_dir = os.path.join(experiment_dir, "ROC_curves")
    roc_plot_opt_png = os.path.join(roc_dir, "roc_optimal_folds_multiclass.png")
    
    # Check if files already exist to skip the whole block
    if os.path.exists(roc_plot_opt_png):
        print(f"\n>>> ROC curves already exist in {roc_dir}. Skipping generation...")
    else:
        print("\nGenerating multiclass ROC curves (One-vs-Rest)...")
        os.makedirs(roc_dir, exist_ok=True)
                
        # Load predictions safely with converters
        df_preds = pd.read_csv(
            preds_filepath,
            converters={
                'y_val': ast.literal_eval,
                'y_prob': ast.literal_eval
            }
        )

        curves_info_optimal = []
        curves_info_median = []

        classifiers = df_resultados["Classifier"].unique()
        all_classes = np.unique([c for sublist in df_preds["y_val"] for c in sublist])
        print(f"Classes found: {all_classes}")

        for clf_name in classifiers:
            df_clf = df_resultados[df_resultados["Classifier"] == clf_name]
            best_fold_idx = df_clf["val_auc"].idxmax()
            best_fold_num = df_clf.loc[best_fold_idx, "Fold"]
            median_auc = df_clf["val_auc"].median()
            median_fold_idx = (df_clf["val_auc"] - median_auc).abs().idxmin()
            median_fold_num = df_clf.loc[median_fold_idx, "Fold"]

            # Optimal fold
            df_clf_preds_best = df_preds[(df_preds["Classifier"] == clf_name) & (df_preds["Fold"] == best_fold_num)]
            if len(df_clf_preds_best) > 0:
                y_val_list_best = df_clf_preds_best.iloc[0]["y_val"]
                y_prob_list_best = df_clf_preds_best.iloc[0]["y_prob"]
                if y_prob_list_best:
                    y_val_bin = label_binarize(y_val_list_best, classes=all_classes)
                    y_prob_arr = np.array(y_prob_list_best)
                    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
                    for i, clase in enumerate(all_classes):
                        fpr, tpr, _ = metrics.roc_curve(y_val_bin[:, i], y_prob_arr[:, i])
                        auc_val = metrics.auc(fpr, tpr)
                        fpr_dict[clase] = fpr
                        tpr_dict[clase] = tpr
                        auc_dict[clase] = auc_val
                    curves_info_optimal.append({"classifier": clf_name, "fold": best_fold_num, "fpr": fpr_dict, "tpr": tpr_dict, "auc": auc_dict})

            # Median fold
            df_clf_preds_median = df_preds[(df_preds["Classifier"] == clf_name) & (df_preds["Fold"] == median_fold_num)]
            if len(df_clf_preds_median) > 0:
                y_val_list_median = df_clf_preds_median.iloc[0]["y_val"]
                y_prob_list_median = df_clf_preds_median.iloc[0]["y_prob"]
                if y_prob_list_median:
                    y_val_bin = label_binarize(y_val_list_median, classes=all_classes)
                    y_prob_arr = np.array(y_prob_list_median)
                    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
                    for i, clase in enumerate(all_classes):
                        fpr, tpr, _ = metrics.roc_curve(y_val_bin[:, i], y_prob_arr[:, i])
                        auc_val = metrics.auc(fpr, tpr)
                        fpr_dict[clase] = fpr
                        tpr_dict[clase] = tpr
                        auc_dict[clase] = auc_val
                    curves_info_median.append({"classifier": clf_name, "fold": median_fold_num, "fpr": fpr_dict, "tpr": tpr_dict, "auc": auc_dict})

        # --- Plot multiclass ROC ---
        n_curvas = len(curves_info_optimal) * len(all_classes)
        palette = sns.color_palette("tab20", n_colors=n_curvas)

        for info, fname in zip([curves_info_optimal, curves_info_median], ["roc_optimal_folds_multiclass.png", "roc_median_folds_multiclass.png"]):
            fig, ax = plt.subplots(figsize=(8, 6))
            c_idx = 0
            for clf_info in info:
                clf_name = clf_info["classifier"]
                fold_num = clf_info["fold"]
                auc_dict = clf_info["auc"]
                for clase in all_classes:
                    fpr = clf_info["fpr"][clase]
                    tpr = clf_info["tpr"][clase]
                    auc_val = auc_dict[clase]
                    color = palette[c_idx % len(palette)]
                    ax.plot(fpr, tpr, label=f"{clf_name} (Fold={fold_num}, Class={clase}, AUC={auc_val:.3f})", color=color)
                    c_idx += 1
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="_nolegend_")
            ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
            ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()
            plt.savefig(os.path.join(roc_dir, fname.replace(".png", ".pdf")), dpi=dpi, bbox_inches='tight')
            plt.savefig(os.path.join(roc_dir, fname), dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"Multiclass ROC plot saved at: {os.path.join(roc_dir, fname)}")
            



# =====================================================================
    # --- Additional analysis execution (complementary scripts) ---
    # =====================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Statistical Comparison (2_model_differences.py)
    model_diff_dir = os.path.join(experiment_dir, "model_differences")
    summary_txt = os.path.join(model_diff_dir, "model_differences_summary.txt") # Archivo que genera el script

    if os.path.exists(summary_txt):
        print(f"\n>>> Statistical comparison already exists in {model_diff_dir}. Skipping...")
    else:
        print("\nExecuting model comparisons (model_differences.py)...")
        os.makedirs(model_diff_dir, exist_ok=True)
        
        postprocess_cmd = [
            "python3", os.path.join(script_dir, "../2_model_differences.py"),
            "--csv_preds", preds_filepath,
            "--csv_results", resultados_filepath,
            "--metric", "val_auc",
            "--alpha", "0.05",
            "--outdir", model_diff_dir
        ]
        subprocess.call(postprocess_cmd)

    # 2. Fine-tuning of best model (3_retrain_best_model_and_evaluate_multiclass.py)
    best_results_dir = os.path.join(experiment_dir, "best_results")
    
    if os.path.exists(os.path.join(best_results_dir, "best_estimator.pkl")):
        print(f"\n>>> Fine-tuning for the best model already exists in {best_results_dir}. Skipping...")
    else:
        # Dynamically identify the best model based on mean val_auc
        mean_auc = df_resultados.groupby("Classifier")["val_auc"].mean()
        best_model_detected = mean_auc.idxmax()
        
        # Name mapping
        model_mapping = {
            "SVM": "SVM",
            "Logistic Regression": "LogisticRegression",
            "Random Forest": "RandomForest",
            "Naive Bayes": "NaiveBayes",
            "KNN": "KNN",
            "Gradient Boosting": "GradientBoosting"
        }
        best_model_finetune = model_mapping.get(best_model_detected, best_model_detected)
        path_features = args.csv    
        print(f"\n>>> Detected best model for Fine-tuning: {best_model_detected}")
        
        # Build command with absolute path
        fine_tune_cmd = [
            "python3", os.path.join(script_dir, "3_retrain_best_model_and_evaluate_multiclass.py"),
            "--csv", path_features,
            "--model", best_model_finetune,
            "--variables", variables_txt_path
        ]
        
        subprocess.call(fine_tune_cmd)

    print(f"\n>>> Pipeline completed. Results in: {experiment_dir}")
