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
from datetime import datetime

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
if __name__ == "__main__":
    """
    Main function orchestrating the full training & evaluation workflow:
    1. Parse CLI arguments
    2. Load & preprocess data
    3. Perform optional feature selection
    4. Train & evaluate models
    5. Generate ROC curves & results
    6. Optionally run auxiliary scripts (model comparisons / fine-tuning)
    """
    # --- CLI argument configuration ---    
    parser = argparse.ArgumentParser(
        description="Model evaluation with repeated stratified group cross-validation"
    )
    parser.add_argument(
        "--csv", type=str, 
        # default="features_t2w_MPfirrmann.csv", 
        help="Name of the features CSV file."
    )
    parser.add_argument(
        "--data_pre", type=str,
        help="Root directory containing radiomics data."
    )
    parser.add_argument(
        "--results_base", type=str,
        help="Base directory where results will be stored."
    )
    parser.add_argument(
        "--label_name", type=str, default="label",
        help="Name of the label column."
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of folds for StratifiedGroupKFold (per repetition)."
    )
    parser.add_argument(
        "--n_repeats", type=int, default=10,
        help="Number of repeated CV cycles."
    )
    parser.add_argument(
        "--feature_strategy", type=str,
        choices=["all", "most_discriminant"],
        default="most_discriminant",
        help="Feature selection strategy: 'all' or 'most_discriminant'."
    )
    parser.add_argument(
        "--calculate_differences", action="store_true", default=True,
        help="If enabled, run model_differences.py."
    )
    parser.add_argument(
        "--fine_tune_best_model", action="store_true", default=False,
        help="If enabled, fine-tune the best model."
    )
    
    args = parser.parse_args()

# --- Gestión de Directorios de Salida (Sistema de Runs) ---
    # Creamos una carpeta 'runs' y dentro una con la fecha y hora actual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_folder = os.path.join(args.results_base, "runs")
    experiment_dir = os.path.join(base_results_folder, f"run_{timestamp}")
    
    os.makedirs(experiment_dir, exist_ok=True)
    print(f">>> Iniciando nuevo experimento en: {experiment_dir}")  
    # --- Data loading & preprocessing ---
    # # path_features = '/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/data/multiclass/features_t2w_MPfirrmann.csv'
    # path_features = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/data/radiomics_final_with_labels.csv"
    
    path_features = os.path.join(args.data_pre, args.csv) 
    print(f"Loading data from: {path_features}")
    df = pd.read_csv(path_features)
    # 1. Limpieza de Pfirrmann: Eliminar nulos reales y valores vacíos (como espacios o '\xa0')
    # Reemplazamos celdas con solo espacios por NaN para poder borrarlas
    label = args.label_name
    df[label] = df[label].replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(subset=[label])

    # 2. Limpieza de Características: Convertir todo a numérico (esto arregla el error \xa0)
    # Identificamos columnas que no son IDs ni etiquetas
    cols_to_exclude = ['patient_id', 'study_id', label]
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    for col in feature_cols:
        # errors='coerce' convierte el '\xa0' o cualquier texto en NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Eliminar filas que quedaron con NaNs en las características (necesario para Shapiro)
    df = df.dropna()

    # 4. Ahora sí, extraemos y y X de un DataFrame limpio
    # Convertimos label a string para que np.unique no falle si hay tipos mixtos
    df[label] = df[label].astype(float).astype(int).astype(str)
    
    y = df[label].values
    groups = df["patient_id"]
    X = df.drop(cols_to_exclude, axis=1)

    experiment_dir = os.path.join(args.results_base, "output_multiclass")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created results folder: {experiment_dir}")


    # --- Feature selection ---
    selected_features = X.columns

    if args.feature_strategy == "most_discriminant":
        print(">> Performing feature selection ...")

        # Directories for feature selection results
        fs_dir = os.path.join(experiment_dir, "feature_selection")
        os.makedirs(fs_dir, exist_ok=True)
        images_dir = os.path.join(fs_dir, f"images")
        os.makedirs(images_dir, exist_ok=True)

        # Initialize lists to store per-feature statistics
        feature_names, test_type_list, pvalue_list = ([] for _ in range(3))

        # Evaluate each feature individually
        for column in X.columns:
            stat, p = shapiro(X[column])
            groups_ = [X[column][y == clase] for clase in np.unique(y)]
            feature_names.append(column)
            
            alpha = 0.05
            if p > alpha:
                test_type_list.append('ANOVA')
                stats, pval = f_oneway(*groups_)
            else:
                test_type_list.append('Kruskal-Wallis')
                stats, pval = kruskal(*groups_)
            pvalue_list.append(pval)

        # DataFrame with all per-feature stats
        full_stats_df = pd.DataFrame(
            list(zip(test_type_list, pvalue_list)),
            index=feature_names,
            columns=['Test', 'p-value']
        ).sort_values(by='p-value', ascending=True)

        # Análisis de Correlación (Filtro de Redundancia)
        print("  --> Removing redundant features (Correlation > 0.85)...")
        
        # Ordenamos X según los p-valores para priorizar las mejores en el filtro
        X_sorted = X[full_stats_df.index]
        corr_matrix = X_sorted.corr(method='spearman').abs() # Spearman es más robusto para radiómica
        
        # Matriz booleana para identificar variables a eliminar
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                # Si la correlación es alta y la variable j no ha sido eliminada
                if corr_matrix.iloc[i, j] > 0.85:
                    colname = corr_matrix.columns[i]
                    to_drop.add(colname)

        # Variables que pasan el filtro de correlación
        non_redundant_features = [f for f in full_stats_df.index if f not in to_drop]
        print(f"  --> {len(to_drop)} features removed due to high correlation")

        print(f"  --> Selected top {len(non_redundant_features)} non-redundant features.")

        # --- Guardar resultados y Heatmap ---
        # Filtrar X a las elegidas
        X = X[non_redundant_features]
        
        # Guardar CSV con los p-valores de las seleccionadas
        full_stats_df.loc[non_redundant_features].to_csv(os.path.join(fs_dir, "selected_features_stats.csv"))
        print(f"  --> Saved CSV with stats for selected features at: {os.path.join(fs_dir, 'selected_features_stats.csv')}")


       # Limit: max 1 feature per 15 samples
        num_features_model = round(X.shape[0] / 15)

        # Solo considerar las no redundantes
        train_df = full_stats_df.loc[non_redundant_features].sort_values(by='p-value', ascending=True)

        final_selected_features = train_df.index[0:num_features_model]

        print(f"  --> Selected {len(final_selected_features)} most relevant features.")

        X = X[final_selected_features]

        # Guardar CSV
        df_path = os.path.join(fs_dir, "final_selected_features_stats.csv")
        train_df.loc[final_selected_features].to_csv(df_path)
        print(f"  --> Saved CSV: {df_path}\n")

        # Generar Heatmap de correlación de las TOP seleccionadas
        plt.figure(figsize=(12, 10))
        sns.heatmap(X.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Spearman Correlation - Selected Features")
        plt.savefig(os.path.join(fs_dir, "correlation_heatmap_final.png"))
        plt.close()
        print(f"  --> Saved correlation heatmap at: {os.path.join(fs_dir, 'correlation_heatmap_final.png')}")

        # --- Generate plots for TOP 20 features ---
        top_20 = train_df.index[:20]

        for rank, feature_name in enumerate(top_20, start=1):
            # Safe filename
            safe_feat_name = feature_name.replace("/", "_")
            feat_folder_name = f"{rank}_{safe_feat_name}"
            feat_folder_path = os.path.join(images_dir, feat_folder_name)
            os.makedirs(feat_folder_path, exist_ok=True)

            # 1. Violin plot by class
            plt.figure(figsize=(9, 9))
            sns.violinplot(x=y, y=df[feature_name], color='grey')
            plt.title(f"Distribution of {feature_name} by class", fontsize=14)
            plt.xlabel("Class")
            plt.ylabel(feature_name)
            violin_plot_path = os.path.join(feat_folder_path, f"{safe_feat_name}_violinplot.png")
            plt.savefig(violin_plot_path, dpi=dpi)
            plt.close()
    else:
        print(">> Using ALL features (no selection).")



    # ##TRAIN AND EVALUATE MODELS

    # Definimos las rutas dentro del experiment_dir (el run actual)
    resultados_filepath = os.path.join(experiment_dir, "metrics.csv")
    preds_filepath = os.path.join(experiment_dir, "predictions.csv")
    variables_txt_path = os.path.join(experiment_dir, "variables_used.txt")

    # --- LÓGICA DE CARGA O ENTRENAMIENTO ---
    if os.path.exists(resultados_filepath) and os.path.exists(preds_filepath):
        print(f"\n>>> The results already exist in {experiment_dir}. Loading files to save time...")

        df_resultados = pd.read_csv(resultados_filepath)
        df_preds = pd.read_csv(
            preds_filepath,
            converters={
                'y_val': ast.literal_eval,
                'y_prob': ast.literal_eval,
                'y_pred': ast.literal_eval
            }
        )
    else:
        print("\n>>> No se encontraron resultados previos. Iniciando entrenamiento de modelos...")
        #--- Model training & evaluation ---
        models = get_models(random_state=42)
        all_results = []
        preds_data = []

        # Evaluate each model
        for model_name, model in models:
            print(f"Evaluating {model_name} ...")
            fold_metrics_list, pred_vals = evaluate_model_multiclass(
                model, X, y, groups,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                base_random_state=42
            )

            for fold_metrics in fold_metrics_list:
                fold_metrics["Classifier"] = model_name
                all_results.append(fold_metrics)

            preds_data.append({
                "Classifier": model_name,
                "folds": pred_vals["folds"]
            })

        # DataFrame con resultados
        df_resultados = pd.DataFrame(all_results)
        fixed_cols = ["Classifier", "Fold", "Repeat"]
        other_cols = [c for c in df_resultados.columns if c not in fixed_cols]
        df_resultados = df_resultados[fixed_cols + other_cols]
        df_resultados.sort_values(by=["Classifier", "Fold"], inplace=True)
        df_resultados.to_csv(resultados_filepath, index=False)
        print(f"Results saved at '{resultados_filepath}'")

        # Preparar y guardar predicciones
        records_for_csv = []
        for item in preds_data:
            clf_name = item["Classifier"]
            for fold_info in item["folds"]:
                records_for_csv.append({
                    "Classifier": clf_name,
                    "Fold": fold_info["fold_index"],
                    "Repeat": fold_info["Repeat"],
                    "y_val": fold_info["y_val"].tolist(),
                    "y_pred": fold_info["y_val_pred"].tolist(),
                    "y_prob": fold_info["y_val_prob"].tolist() if fold_info["y_val_prob"] is not None else []
                })

        df_preds = pd.DataFrame(records_for_csv)
        df_preds.to_csv(preds_filepath, index=False)
        print(f"Predictions saved at '{preds_filepath}'")

        # Guardar lista de variables usadas
        with open(variables_txt_path, "w") as f:
            for feat in selected_features:
                f.write(str(feat) + "\n")
        print(f"Variables file saved at: {variables_txt_path}")

    # --- GENERACIÓN DE CURVAS ROC (Usando el experiment_dir dinámico) ---
    print("\nGenerating multiclass ROC curves (One-vs-Rest)...")
    
    roc_dir = os.path.join(experiment_dir, "ROC_curves")
    os.makedirs(roc_dir, exist_ok=True)

    curves_info_optimal = []
    curves_info_median = []

    classifiers = df_resultados["Classifier"].unique()
    all_classes = np.unique(df_preds["y_val"].explode())
    print(f"Classes found: {all_classes}")

    for clf_name in classifiers:
        df_clf = df_resultados[df_resultados["Classifier"] == clf_name]
        
        # Fold óptimo
        best_fold_idx = df_clf["val_auc"].idxmax()
        best_fold_num = df_clf.loc[best_fold_idx, "Fold"]
        
        # Fold mediano
        median_auc = df_clf["val_auc"].median()
        median_fold_idx = (df_clf["val_auc"] - median_auc).abs().idxmin()
        median_fold_num = df_clf.loc[median_fold_idx, "Fold"]

        # Extraer curvas
        for fold_num, target_list in zip([best_fold_num, median_fold_num], [curves_info_optimal, curves_info_median]):
            df_clf_preds = df_preds[(df_preds["Classifier"] == clf_name) & (df_preds["Fold"] == fold_num)]
            
            if len(df_clf_preds) > 0:
                y_val_list = df_clf_preds.iloc[0]["y_val"]
                y_prob_list = df_clf_preds.iloc[0]["y_prob"]
                
                if y_prob_list:
                    y_val_bin = label_binarize(y_val_list, classes=all_classes)
                    y_prob_arr = np.array(y_prob_list)
                    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
                    
                    for i, clase in enumerate(all_classes):
                        fpr, tpr, _ = metrics.roc_curve(y_val_bin[:, i], y_prob_arr[:, i])
                        auc_val = metrics.auc(fpr, tpr)
                        fpr_dict[clase] = fpr
                        tpr_dict[clase] = tpr
                        auc_dict[clase] = auc_val
                        
                    target_list.append({
                        "classifier": clf_name,
                        "fold": fold_num,
                        "fpr": fpr_dict,
                        "tpr": tpr_dict,
                        "auc": auc_dict
                    })

    # Plotting
    for info, fname in zip([curves_info_optimal, curves_info_median], ["roc_optimal_folds_multiclass.png", "roc_median_folds_multiclass.png"]):
        if not info: continue
        fig, ax = plt.subplots(figsize=(10, 8))
        palette = sns.color_palette("tab20", n_colors=len(info) * len(all_classes))
        color_idx = 0
        
        for clf_info in info:
            for clase in all_classes:
                fpr = clf_info["fpr"][clase]
                tpr = clf_info["tpr"][clase]
                auc_v = clf_info["auc"][clase]
                ax.plot(fpr, tpr, label=f"{clf_info['classifier']} (Fold {clf_info['fold']}, Cl {clase}, AUC={auc_v:.3f})", color=palette[color_idx])
                color_idx += 1
        
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(roc_dir, fname), dpi=dpi)
        plt.close()

    # --- RESULTADOS FINALES Y SCRIPTS EXTERNOS ---
    mean_auc = df_resultados.groupby("Classifier")["val_auc"].mean().sort_values(ascending=False)
    print(f"\nRanking de modelos (Mean AUC):\n{mean_auc}")

    # Ejecutar comparación estadística
    print("\nRunning model comparisons (model_differences.py) ...")
    import subprocess
    model_diff_dir = os.path.join(experiment_dir, "model_differences")
    os.makedirs(model_diff_dir, exist_ok=True)

    subprocess.call([
        "python3", "2_model_differences.py",
        "--csv_preds", preds_filepath,
        "--csv_results", resultados_filepath,
        "--metric", "val_auc",
        "--outdir", model_diff_dir
    ])

    # Fine-tuning del mejor modelo
    best_model_name = mean_auc.idxmax()
    model_mapping = {
        "SVM": "SVM", "Logistic Regression": "LogisticRegression",
        "Random Forest": "RandomForest", "Naive Bayes": "NaiveBayes",
        "KNN": "KNN", "Gradient Boosting": "GradientBoosting"
    }
    best_model_key = model_mapping.get(best_model_name, "GradientBoosting")

    print(f"\nFine-tuning best model: {best_model_key}")
    subprocess.call([
        "python3", "3_retrain_best_model_and_evaluate_multiclass.py",
        "--csv", path_features,
        "--model", best_model_key,
        "--variables", variables_txt_path,
        "--label", args.label_name,
        "--experiment_dir", experiment_dir
    ])
