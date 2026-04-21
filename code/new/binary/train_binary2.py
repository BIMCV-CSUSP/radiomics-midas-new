import argparse
import pandas as pd
import numpy as np
import os
import ast


from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn import metrics

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LassoCV
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
from datetime import datetime
plt.style.use(['science', 'grid'])
dpi = 300
from scipy.stats import kruskal, f_oneway
plt.rcParams["text.usetex"] = False
import subprocess

def get_models(random_state=42):
    """
    Define pipelines for each classifier, including standard preprocessing.
    
    Args:
        random_state (int): Seed for reproducibility
    
    Returns:
        list: List of tuples (model_name, scikit_pipeline)
    """

    # Pipeline for Support Vector Machine
    pipe_svc = make_pipeline(
        StandardScaler(), # Feature normalization
        VarianceThreshold(),  # Remove features with null variance
        SVC(random_state=random_state, class_weight="balanced", probability=True)
    )
    
    # Pipeline for Logistic Regression
    pipe_lr = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        LogisticRegression(
            penalty='elasticnet',       # Combined L1 and L2 regularization
            l1_ratio=0.5,               # Ratio for elasticnet (0.5 = equal weight L1 and L2)
            class_weight="balanced",
            random_state=random_state,
            solver='saga',              # Optimizer for elasticnet
            max_iter=10000              # Maximum iterations
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
        GaussianNB() # No additional parameters needed
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

    # List with all models
    models = [
        ("SVM", pipe_svc),
        ("Logistic Regression", pipe_lr),
        ("Random Forest", pipe_rf),
        ("Naive Bayes", pipe_nb),
        ("KNN", pipe_knn),
        ("Gradient Boosting", pipe_gb),
    ]
    return models


def evaluate_model(model, X, y, groups, n_splits=5, n_repeats=1, base_random_state=42):
    """
    Performs repeated stratified cross-validation by groups (patients).
    
    Args:
        model: Model to evaluate (scikit-learn pipeline)
        X (pd.DataFrame): Features
        y (np.array): Binary labels (0/1)
        groups (np.array): Group identifiers (patients) for CV
        n_splits (int): Number of partitions per repetition
        n_repeats (int): Number of cross-validation repetitions
        base_random_state (int): Base seed for reproducibility
    
    Returns:
        tuple: (fold_results, pred_vals)
            - fold_results: List of dictionaries with metrics per fold
            - pred_vals: Dict with prediction data for each fold
    """

    fold_results = []   # List to store metrics for each fold
    folds_data = []     # List to store prediction data

    global_fold_index = 0
    for rep in range(n_repeats):
        # Each repetition uses a different seed to get different partitions
        current_random_state = base_random_state + rep
        
        # StratifiedGroupKFold ensures similar class distribution
        # while maintaining separation of groups (patients) between train/val
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=current_random_state
        )
        
        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            global_fold_index += 1
            
            # Split data into training and validation
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # --- Metrics on training set ---
            y_train_pred = model.predict(X_train)

            # Get probabilities or decision scores if available
            if hasattr(model, "predict_proba"):
                y_train_prob = model.predict_proba(X_train)[:, 1]
            elif hasattr(model, "decision_function"):
                y_train_prob = model.decision_function(X_train)
            else:
                y_train_prob = None
            
            # Calculate AUC and F1 on training
            try:
                train_auc = roc_auc_score(y_train, y_train_prob) if y_train_prob is not None else np.nan
            except:
                train_auc = np.nan
            train_f1 = f1_score(y_train, y_train_pred, average="binary")
            

            # --- Metrics on validation set ---
            y_val_pred = model.predict(X_val)
            
            # Get probabilities for validation
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                y_val_prob = model.decision_function(X_val)
            else:
                y_val_prob = None
            
            # Calculate AUC on validation
            try:
                val_auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else np.nan
            except:
                val_auc = np.nan
            
            # Complete performance metrics on validation
            val_mcc = matthews_corrcoef(y_val, y_val_pred)          # Matthews correlation coefficient
            val_kappa = cohen_kappa_score(y_val, y_val_pred)        # Cohen's Kappa (vs chance)
            val_f1_binary = f1_score(y_val, y_val_pred, average="binary")  # Binary F1
            val_f1_macro = f1_score(y_val, y_val_pred, average="macro")    # Macro F1
            val_accuracy = accuracy_score(y_val, y_val_pred)               # Accuracy
            val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)  # Balanced accuracy
            val_sensitivity = recall_score(y_val, y_val_pred, pos_label=1)      # Sensitivity
            val_specificity = recall_score(y_val, y_val_pred, pos_label=0)      # Specificity
            val_ppv = precision_score(y_val, y_val_pred, pos_label=1)           # Positive predictive value
            
            # Confusion matrix for additional calculations
            cm = confusion_matrix(y_val, y_val_pred)

            # Calculate negative predictive value (NPV)
            if (cm[0, 0] + cm[1, 0]) > 0:
                val_npv = cm[0, 0] / (cm[0, 0] + cm[1, 0])
            else:
                val_npv = np.nan
            
            # Per-class metrics
            per_class_precision = precision_score(y_val, y_val_pred, average=None)
            per_class_recall = recall_score(y_val, y_val_pred, average=None)
            per_class_f1 = f1_score(y_val, y_val_pred, average=None)
            
            # Per-class accuracy (diagonal of normalized matrix by rows)
            per_class_accuracy = []
            for i in range(len(cm)):
                row_sum = np.sum(cm[i, :])
                if row_sum > 0:
                    per_class_accuracy.append(cm[i, i] / row_sum)
                else:
                    per_class_accuracy.append(np.nan)
            
             # Collect all metrics in a dictionary
            fold_metrics = {
                "Fold": global_fold_index,  
                "Repeat": rep + 1,          
                "train_auc": train_auc,
                "train_f1": train_f1,
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
                "per_class_accuracy": per_class_accuracy
            }
            
            fold_results.append(fold_metrics)
    
            # Save data from this fold for later analysis (ROC curves, etc.)
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


if __name__ == "__main__":
    """
    Main function that coordinates the complete training and evaluation process:
    1. Process command line arguments
    2. Load and preprocess data
    3. Perform feature selection (optional)
    4. Train and evaluate models
    5. Generate ROC curves and results
    6. Execute complementary scripts (optional)
    """
    # --- Command line arguments configuration ---    
    parser = argparse.ArgumentParser(description="Automatización Binaria Completa")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--results_base", type=str, default="/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/data/binary_new")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--resume_run", type=str, default=None, 
                    help="Ruta de una carpeta de 'run' previa para reusar resultados (ej: run_bin_20260225_114415)")
    args = parser.parse_args()


    # ==============================================================================
    # BLOQUE 1: CARGA Y LIMPIEZA DE DATOS (ESTRATEGIA ANTI-DATA LEAKAGE)
    # ==============================================================================

    # # --- Data loading and preprocessing ---
    # --- 1. Gestión de Run ---
    if args.resume_run:
        # Si le pasamos un nombre de carpeta, la usamos
        experiment_dir = os.path.join(args.results_base, "runs", args.resume_run)
        print(f">>> Reutilizando experimento: {experiment_dir}")
    else:
        # Si no, creamos una nueva como tenías
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(args.results_base, "runs", f"run_bin_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f">>> Iniciando nuevo experimento en: {experiment_dir}")

    # Las rutas a los archivos ahora serán relativas a experiment_dir
    fs_dir = os.path.join(experiment_dir, "feature_selection")
    df_path_1 = os.path.join(fs_dir, "train_auc_pvals_df.csv")

    # # ==============================================================================
    # # Bloque 2: Selección de Características (Feature Selection) 
    # # ==============================================================================
    
    # # --- 2. Cleaning (Avoid Leakage) ---
    # df = pd.read_csv(args.csv)
    # y = df["label"].values
    # groups = df["patient_id"].values
    # # Identificar columnas a eliminar (Leakage Prevention)
    # cols_to_drop = ['id_igtp','patient_id', 'study_id', 'label', 'mask_type','SSA_type']
    # X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    # # Eliminar columnas de diagnóstico (metadatos de radiómica)
    # X = X.drop(columns=[c for c in X.columns if 'diagnostics' in c], errors='ignore')

    # # --- Feature selection ---
    # selected_features = X.columns
    # if os.path.exists(df_path_1):
    #     print(f">> Cargando selección de variables existente desde: {df_path_1}")
    #     train_auc_pvals_df = pd.read_csv(df_path_1, index_col=0)
    #     # Seleccionar las N mejores basadas en el rendimiento del archivo cargado
    #     num_features_model = round(X.shape[0]/10)
    #     selected_features = train_auc_pvals_df.index[0:num_features_model].tolist()
    #     X = X[selected_features]
    # else:
    #     print(">> No se encontró análisis previo. Iniciando selección de características...")
    #     os.makedirs(fs_dir, exist_ok=True)
    #     images_dir = os.path.join(fs_dir, "images")
    #     os.makedirs(images_dir, exist_ok=True)

    #     # Initialize lists to store statistics per feature
    #     feature_names, sensitivity_list, specificity_list = ([] for _ in range(3))
    #     auc_list, threshold_list, test_type_list, pvalue_list, pos_vs_neg_list = ([] for _ in range(5))
        
    #     # Evaluate each feature individually
    #     for column in X.columns:
    #         # Shapiro-Wilk normality test
    #         stat, p = shapiro(X[column])
            
    #         # Get distributions by class
    #         a_dist = X[column][y == 0]  # Class 0
    #         b_dist = X[column][y == 1]  # Class 1
            
    #         feature_names.append(column)
            
    #         # Select statistical test according to normality
    #         alpha = 0.05
    #         if p > alpha: # If p > 0.05, assume normality
    #             test_type_list.append('t-test')
    #             _, pval = ttest_ind(a_dist, b_dist) # T-test for normal data
    #         else:
    #             test_type_list.append('mann-whitney U-test')
    #             _, pval = mannwhitneyu(a_dist, b_dist) # Non-parametric test
    #         pvalue_list.append(pval)
            
    #         # Evaluate discriminative capacity (AUC)
    #         fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=1)
    #         auc_val = metrics.auc(fpr, tpr)

    #         # If AUC < 0.5, invert the relationship (greater/lesser)
    #         pos_vs_neg = ">" 
    #         if auc_val < 0.5:
    #             fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=0)
    #             auc_val = metrics.auc(fpr, tpr)
    #             pos_vs_neg = "<"
    #         auc_list.append(auc_val)
    #         pos_vs_neg_list.append(pos_vs_neg)
            
    #         # Find optimal point in ROC curve (Youden's J index)
    #         roc_df = pd.DataFrame({
    #             'fpr': fpr,
    #             'tpr': tpr,
    #             '1-fpr': 1 - fpr,
    #             'tf': tpr - (1 - fpr),   # Youden's J = Sensitivity + Specificity - 1
    #             'thresholds': thresholds
    #         })
    #         cutoff_df = roc_df.iloc[(roc_df.tf - 0).abs().argsort()[:1]] # Closest point to optimal
            
    #         # Save sensitivity, specificity and optimal threshold
    #         sensitivity_list.append(cutoff_df['tpr'].values[0])
    #         specificity_list.append(cutoff_df['1-fpr'].values[0])
    #         threshold_list.append(cutoff_df['thresholds'].values[0])
        
    #     # Create DataFrame with all statistics per feature
    #     train_auc_pvals_df = pd.DataFrame(
    #         list(zip(auc_list, pos_vs_neg_list, threshold_list,
    #                     sensitivity_list, specificity_list, 
    #                     test_type_list, pvalue_list)),
    #         index=feature_names,
    #         columns=['AUC', 'Pos.vs.Neg.', 'Cutoff-Threshold', 'Sensitivity',
    #                     'Specificity', 'Test', 'p-value']
    #     ).sort_values(by='p-value', ascending=True) #

    #     df_path_raw = os.path.join(fs_dir, "ranking_puro_sin_correlacion.csv")
    #     train_auc_pvals_df.to_csv(df_path_raw)
    #     print(f"  --> Guardado ranking inicial (sin filtro): {df_path_raw}")

    #     # --- FILTRO DE REDUNDANCIA (CORRELACIÓN) ---
    #     print("  --> Eliminando características redundantes (Spearman > 0.85)...")
    #     X_sorted = X[train_auc_pvals_df.index]
    #     corr_matrix = X_sorted.corr(method='spearman').abs()

    #     to_drop = set()
    #     for i in range(len(corr_matrix.columns)):
    #         for j in range(i):
    #             if corr_matrix.iloc[i, j] > 0.85:
    #                 colname = corr_matrix.columns[i]
    #                 to_drop.add(colname)

    #     # Aplicar limpieza
    #     train_df_filtered = train_auc_pvals_df.drop(index=list(to_drop))
    #     print(f"  --> {len(to_drop)} variables eliminadas por alta correlación.")


    #     # Select features: maximum 1 feature per 15 samples
    #     num_features_model = round(X.shape[0]/10)
    #     train_df = train_df_filtered.sort_values(by='p-value', ascending=True)

    #     # Select the N most significant features
    #     selected_features = train_df.index[0:num_features_model]
    #     print(f"  --> Selected {len(selected_features)} most relevant features.")

    #     # Filter DataFrame to use only selected features
    #     X = X[selected_features]
    #     # Save DataFrame with complete statistics
    #     df_path_1 = os.path.join(fs_dir, f"train_auc_pvals_df.csv")
    #     train_df_filtered.loc[selected_features].to_csv(df_path_1)
    #     print(f"  --> Saved CSV: {df_path_1}\n")

    #     top_20 = train_df_filtered.index[:20]


    #     for rank, feature_name in enumerate(top_20, start=1):
    #         # Create filename
    #         safe_feat_name = feature_name.replace("/", "_")
    #         feat_folder_name = f"{rank}_{safe_feat_name}"
    #         feat_folder_path = os.path.join(images_dir, feat_folder_name)
    #         os.mkdir(feat_folder_path)
            
    #         # 1. Violin plot to visualize distributions by class
    #         plt.figure(figsize=(9, 9))
    #         sns.violinplot(x=y, y=df[feature_name], color='grey')
    #         plt.title(f"Distribution of {feature_name} in 0 vs 1", fontsize=14)
    #         plt.xlabel("Classes")
    #         plt.xticks([0, 1], ["0", "1"], fontsize=12)
    #         violin_plot_path = os.path.join(feat_folder_path, f"{safe_feat_name}_violinplot.png")
    #         plt.savefig(violin_plot_path, dpi=dpi)
    #         plt.close()
            
        
    # #correlation matrix of the features only with the first 20 features
    # if len(selected_features) > 1:
    #     features_for_corr = X.columns[:200]
    #     corr_matrix = X[features_for_corr].corr()
    #     plt.figure(figsize=(12, 10))
    #     sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    #     plt.title("Features Correlation Matrix", fontsize=16)
    #     plt.xticks(rotation=45, ha='right')
    #     plt.tight_layout()
    #     corr_plot_path = os.path.join(experiment_dir, "correlation_matrix.png")
    #     plt.savefig(corr_plot_path, dpi=dpi)
    #     print(f"  --> Saved correlation matrix: {corr_plot_path}\n")
        

    # ==============================================================================
    # Bloque 2: Selección de Características (Estrategia LASSO Multivariante)
    # ==============================================================================
    
    # --- 2. Cleaning (Avoid Leakage) ---
    df = pd.read_csv(args.csv)
    y = df["label"].values
    groups = df["patient_id"].values
    
    cols_to_drop = ['id_igtp','patient_id', 'study_id', 'label', 'mask_type','SSA_type']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    X = X.drop(columns=[c for c in X.columns if 'diagnostics' in c], errors='ignore')

    # --- Feature selection ---
    if os.path.exists(df_path_1):
        print(f">> Cargando selección de variables existente desde: {df_path_1}")
        train_df = pd.read_csv(df_path_1, index_col=0)
        selected_features = train_df.index.tolist()
        X = X[selected_features]
    else:
        print(">> No se encontró análisis previo. Iniciando selección multivariante (LASSO)...")
        os.makedirs(fs_dir, exist_ok=True)
        images_dir = os.path.join(fs_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 1. Análisis Univariante Inicial (Para tener las estadísticas base)
        from scipy.stats import shapiro, ttest_ind, mannwhitneyu
        from sklearn import metrics
        
        feature_names, pvalue_list, auc_list, test_type_list = [], [], [], []
        pos_vs_neg_list, threshold_list, sens_list, spec_list = [], [], [], []

        for column in X.columns:
            stat, p_norm = shapiro(X[column])
            a_dist = X[column][y == 0]
            b_dist = X[column][y == 1]
            
            if p_norm > 0.05:
                test_type = 't-test'
                _, pval = ttest_ind(a_dist, b_dist)
            else:
                test_type = 'mann-whitney U-test'
                _, pval = mannwhitneyu(a_dist, b_dist)
            
            fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=1)
            auc_val = metrics.auc(fpr, tpr)
            pos_vs_neg = ">"
            if auc_val < 0.5:
                fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=0)
                auc_val = metrics.auc(fpr, tpr)
                pos_vs_neg = "<"

            # Youden Index para Cutoff
            j_idx = tpr - fpr
            best_idx = np.argmax(j_idx)
            
            feature_names.append(column)
            pvalue_list.append(pval)
            auc_list.append(auc_val)
            test_type_list.append(test_type)
            pos_vs_neg_list.append(pos_vs_neg)
            threshold_list.append(thresholds[best_idx])
            sens_list.append(tpr[best_idx])
            spec_list.append(1 - fpr[best_idx])

        # Ranking base (sin filtro aún)
        train_auc_pvals_df = pd.DataFrame({
            'AUC': auc_list, 'Pos.vs.Neg.': pos_vs_neg_list, 'Cutoff-Threshold': threshold_list,
            'Sensitivity': sens_list, 'Specificity': spec_list, 'Test': test_type_list, 'p-value': pvalue_list
        }, index=feature_names)

        # 2. Filtro de Redundancia (Spearman > 0.85)
        X_numeric = X.select_dtypes(include=[np.number])
        corr_matrix = X_numeric.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        X_filtered = X_numeric.drop(columns=to_drop)
        print(f"  --> {len(to_drop)} variables eliminadas por alta correlación (>0.85).")

        # # 3. Selección Multivariante con LASSO

        
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X_filtered)
        
        # # Intentamos LASSO
        # lasso = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_scaled, y)
        # coef = pd.Series(lasso.coef_, index=X_filtered.columns)
        # selected_lasso = coef[coef != 0].abs().sort_values(ascending=False)

        # # --- CLÁUSULA DE SEGURIDAD ---
        # num_max = max(3, int(X.shape[0] / 10))
        
        # if len(selected_lasso) >= 3:
        #     print(f"  --> LASSO ha funcionado. Seleccionando las top {num_max} variables.")
        #     selected_features = selected_lasso.index[:num_max].tolist()
        # else:
        #     print("  --> ADVERTENCIA: LASSO ha sido demasiado estricto (0 o pocas variables).")
        #     print(f"  --> Rescatando las {num_max} mejores variables por p-valor individual.")
        #     # Rescatamos del ranking univariante que calculamos al principio del bloque
        #     ranking_filtrado = train_auc_pvals_df.loc[X_filtered.columns]
        #     selected_features = ranking_filtrado.sort_values(by='p-value').index[:num_max].tolist()
        
        # # 4. Preparación de X y train_df final
        # X = X[selected_features]
        # train_df = train_auc_pvals_df.loc[selected_features].copy()


        # 3. Selección Multivariante: RFE como motor principal (LASSO opcional)
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import RFECV

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        print("  --> Iniciando RFECV (Búsqueda del set óptimo)...")
        
        # Usamos un modelo base robusto para RFE
        # Ponemos C=1.0 para que no sea tan estricto como LassoCV
        base_model = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            class_weight='balanced', 
            random_state=42,
            C=1.0 # Menos penalización para dejar pasar señal
        )

        # El selector buscará el número de variables (mínimo 2, máximo 10) que mejor F1 den
        selector = RFECV(
            estimator=base_model,
            step=1,
            cv=5, 
            scoring='f1',
            min_features_to_select=2
        )
        
        selector.fit(X_scaled, y)
        
        # Extraemos las supervivientes
        selected_features = X_filtered.columns[selector.support_].tolist()

        # --- CLÁUSULA DE RESCATE (Si RFE también es muy estricto) ---
        if len(selected_features) < 2:
            print(" --> ADVERTENCIA: RFE demasiado estricto. Rescatando Top 5 por AUC/p-valor.")
            ranking_filtrado = train_auc_pvals_df.loc[X_filtered.columns]
            selected_features = ranking_filtrado.sort_values(by='p-value').index[:5].tolist()
        else:
            print(f" --> RFE ha encontrado {len(selected_features)} variables óptimas.")

        # 4. Preparación de X y train_df final
        X = X[selected_features]
        train_df = train_auc_pvals_df.loc[selected_features].copy()
        
        # Añadimos el peso de Lasso si existe, si no, ponemos 0
        train_df['Lasso_Weight'] = [selected_lasso.get(f, 0) for f in selected_features]
        
        # Guardar CSV final
        df_path_1 = os.path.join(fs_dir, "train_auc_pvals_df.csv")
        train_df.to_csv(df_path_1)
        print(f"  --> Selección final confirmada: {selected_features}")

        # --- GENERACIÓN DE IMÁGENES (Solo las seleccionadas) ---
        for rank, feature_name in enumerate(selected_features, start=1):
            safe_feat_name = feature_name.replace("/", "_")
            feat_folder_path = os.path.join(images_dir, f"{rank}_{safe_feat_name}")
            os.makedirs(feat_folder_path, exist_ok=True)
            
            plt.figure(figsize=(9, 9))
            sns.violinplot(x=y, y=df[feature_name], color='grey')
            plt.title(f"Distribución: {feature_name}\n(Lasso Weight: {train_df.loc[feature_name, 'Lasso_Weight']:.4f})")
            plt.savefig(os.path.join(feat_folder_path, f"{safe_feat_name}_violinplot.png"), dpi=dpi)
            plt.close()

    # Bloque final de correlación (Matriz de las seleccionadas)
    if len(selected_features) > 1:
        corr_matrix_final = X.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix_final, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Matriz de Correlación de la Firma Seleccionada")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "correlation_matrix_final.png"), dpi=dpi)    

    # ==============================================================================
    # Bloque 3: Entrenamiento y Evaluación de Modelos (Cross-Validation)  
    # ==============================================================================
    results_filepath = os.path.join(experiment_dir, "results.csv")
    preds_filepath = os.path.join(experiment_dir, "predictions.csv")
    variables_txt_path = os.path.join(experiment_dir, "variables_used.txt")
    with open(variables_txt_path, "w") as f:
        for feat in selected_features:
            f.write(str(feat) + "\n")
    print(f"File with used variables saved at: {variables_txt_path}")


    if os.path.exists(results_filepath) and os.path.exists(preds_filepath):
            print(f"\n>>> Resultados encontrados en {experiment_dir}. Cargando para análisis...")
            df_results = pd.read_csv(results_filepath)
            df_preds = pd.read_csv(
                preds_filepath, 
                converters={
                    'y_val': ast.literal_eval, 
                    'y_pred': ast.literal_eval, 
                    'y_prob': ast.literal_eval
                }
            )

    else:
        print("\n>>> No se detectaron resultados previos. Iniciando entrenamiento...")
        models = get_models(random_state=42)

        all_results = []
        preds_data = []

        # Evaluate each model
        for model_name, model in models:
            print(f"Evaluating {model_name}...")
            fold_metrics_list, pred_vals = evaluate_model(
                model, X, y, groups,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                base_random_state=42
            )

            # Add classifier name to each result
            for fold_metrics in fold_metrics_list:
                fold_metrics["Classifier"] = model_name
                all_results.append(fold_metrics)

            # Store predictions
            preds_data.append({
                "Classifier": model_name,
                "folds": pred_vals["folds"]
            })

        # Create DataFrame with all results
        df_results = pd.DataFrame(all_results)
        fixed_cols = ["Classifier", "Fold", "Repeat"]
        other_cols = [c for c in df_results.columns if c not in fixed_cols]
        df_results = df_results[fixed_cols + other_cols]
        df_results.sort_values(by=["Classifier", "Fold"], inplace=True)
        
        df_results.to_csv(results_filepath, index=False)
        print(f"Results saved at '{results_filepath}'")



        records_for_csv = []
        for item in preds_data:
            clf_name = item["Classifier"]
            for fold_info in item["folds"]:
                # Convertir arrays de numpy a listas para compatibilidad con CSV/JSON
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


    # --- ROC curves generation ---
    roc_dir = os.path.join(experiment_dir, "ROC_curves")
    roc_plot_path_opt = os.path.join(roc_dir, "roc_optimal_folds.png")
    roc_plot_path_med = os.path.join(roc_dir, "roc_median_folds.png")
    # Collectors for ROC curve information
    curves_info_optimal = []  # For fold with best AUC of each model
    curves_info_median = []   # For fold with median AUC of each model
    # Automatización: Solo generar si no existen los archivos finales
    if os.path.exists(roc_plot_path_opt) and os.path.exists(roc_plot_path_med):
        print(f">>> Gráficos ROC ya existentes en {roc_dir}. Saltando generación...")
    else:
        print("\nGenerating ROC curves: optimal and median fold per classifier...")
        os.makedirs(roc_dir, exist_ok=True)

        

        # Process each classifier
        classifiers = df_results["Classifier"].unique()
        for clf_name in classifiers:
            df_clf = df_results[df_results["Classifier"] == clf_name]
            
            # --- Identify optimal fold (best AUC) ---
            best_fold_idx = df_clf["val_auc"].idxmax()
            best_fold_num = df_clf.loc[best_fold_idx, "Fold"]
            
            # --- Identify median fold (AUC closest to median) ---
            median_auc = df_clf["val_auc"].median()
            median_fold_idx = (df_clf["val_auc"] - median_auc).abs().idxmin()
            median_fold_num = df_clf.loc[median_fold_idx, "Fold"]
            
            # --- Process data for optimal fold ---
            df_clf_preds_best = df_preds[
                (df_preds["Classifier"] == clf_name) & 
                (df_preds["Fold"] == best_fold_num)
            ]
            
            if len(df_clf_preds_best) > 0:
                y_val_list_best = df_clf_preds_best.iloc[0]["y_val"]
                y_prob_list_best = df_clf_preds_best.iloc[0]["y_prob"]
                if y_prob_list_best:
                    fpr_best, tpr_best, _ = metrics.roc_curve(y_val_list_best, y_prob_list_best, pos_label=1)
                    auc_val_best = metrics.auc(fpr_best, tpr_best)
                    curves_info_optimal.append({
                        "classifier": clf_name,
                        "fold": best_fold_num,
                        "fpr": fpr_best,
                        "tpr": tpr_best,
                        "auc": auc_val_best
                    })
            
            # --- Process data for median fold ---
            df_clf_preds_median = df_preds[
                (df_preds["Classifier"] == clf_name) & 
                (df_preds["Fold"] == median_fold_num)
            ]
            
            if len(df_clf_preds_median) > 0:
                y_val_list_median = df_clf_preds_median.iloc[0]["y_val"]
                y_prob_list_median = df_clf_preds_median.iloc[0]["y_prob"]
                if y_prob_list_median:
                    fpr_median, tpr_median, _ = metrics.roc_curve(y_val_list_median, y_prob_list_median, pos_label=1)
                    auc_val_median = metrics.auc(fpr_median, tpr_median)
                    curves_info_median.append({
                        "classifier": clf_name,
                        "fold": median_fold_num,
                        "fpr": fpr_median,
                        "tpr": tpr_median,
                        "auc": auc_val_median
                    })

        # Sort curves of each type by descending AUC
        curves_info_optimal.sort(key=lambda x: x["auc"], reverse=True)
        curves_info_median.sort(key=lambda x: x["auc"], reverse=True)

        # Paleta de colores para consistencia visual
        my_colors = ["#0072B2", "#009E73", "#D55E00", "#CC78BC", "#DE8F05", "#56B4E9"]
        my_palette = sns.color_palette(my_colors)
        fixed_classifiers = ["SVM", "Logistic Regression", "Random Forest", "Naive Bayes", "KNN", "Gradient Boosting"]
        color_mapping = {clf: my_palette[i] for i, clf in enumerate(fixed_classifiers)}

        # --- Generate ROC plot for optimal folds ---
        fig_opt, ax_opt = plt.subplots(figsize=(8, 6))
        for info in curves_info_optimal:
            ax_opt.plot(info["fpr"], info["tpr"], label=f"{info['classifier']} (Fold={info['fold']}, AUC={info['auc']:.3f})", 
                        color=color_mapping[info['classifier']])

        ax_opt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="_nolegend_")
        ax_opt.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
        ax_opt.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
        ax_opt.legend(fontsize=10)
        fig_opt.tight_layout()
        
        plt.savefig(roc_plot_path_opt, dpi=dpi, bbox_inches='tight')
        plt.savefig(roc_plot_path_opt.replace(".png", ".pdf"), dpi=dpi, bbox_inches='tight')
        plt.close(fig_opt)

        # --- Generate ROC plot for median folds ---
        fig_med, ax_med = plt.subplots(figsize=(8, 6))
        for info in curves_info_median:
            ax_med.plot(info["fpr"], info["tpr"], label=f"{info['classifier']} (Fold={info['fold']}, AUC={info['auc']:.3f})", 
                        color=color_mapping[info['classifier']])
            
        ax_med.plot([0, 1], [0, 1], linestyle='--', color='gray', label="_nolegend_")
        ax_med.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
        ax_med.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
        ax_med.legend(fontsize=10)
        fig_med.tight_layout()

        plt.savefig(roc_plot_path_med, dpi=dpi, bbox_inches='tight')
        plt.savefig(roc_plot_path_med.replace(".png", ".pdf"), dpi=dpi, bbox_inches='tight')
        plt.close(fig_med)
        print(f"ROC plots saved in: {roc_dir}")


    # ==============================================================================
    #     Bloque 4: Comparación Estadística de Modelos 
    # ==============================================================================


    # 1. Ejecución de comparación estadística (model_differences.py)
    model_diff_dir = os.path.join(experiment_dir, "model_differences")
    summary_txt = os.path.join(model_diff_dir, "model_differences_summary.txt")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.exists(summary_txt):
        print(f"\n>>> Comparación estadística ya existente en {model_diff_dir}. Saltando...")
    else:
        print("\nExecuting model comparisons (model_differences.py)...")
        # Asegurar creación de directorio (makedirs evita error si existe parcial)
        os.makedirs(model_diff_dir, exist_ok=True)

        postprocess_cmd = [
            "python3", os.path.join(script_dir, "../2_model_differences.py"), ###VERIFICACION PREVIA AL FINE TUNING, PARA COMPROBAR QUE LA MEAN AUC ES REALMENTE MEJOR EN EL MEJOR MODELO DETECTADO
            "--csv_preds", preds_filepath,  
            "--csv_results", results_filepath, 
            "--metric", "val_auc",  
            "--alpha", "0.05",  
            "--outdir", model_diff_dir  
        ]
        subprocess.call(postprocess_cmd)

    # ==============================================================================
    #     Bloque 5: Optimización y Explicabilidad 
    # ==============================================================================

    # 2. Fine-tuning y re-entrenamiento del mejor modelo (Script 3)
    best_results_dir = os.path.join(experiment_dir, "best_results") # Carpeta que crea el Script 3
    
    if os.path.exists(best_results_dir):
        print(f"\n>>> El re-entrenamiento del mejor modelo ya existe en {best_results_dir}. Saltando...")
    else:
        # # Fine-tuning del mejor modelo
        # if len(curves_info_optimal) > 0:
        #     best_model = curves_info_optimal[0]["classifier"]
        #     print("The best model is:", best_model)
        # else:
        #     # Fallback si no hay curvas (ej. si se cargaron resultados pero no se regeneraron las curvas)
        #     best_model = df_results.groupby("Classifier")["val_auc"].mean().idxmax()
        
        # ----
        print("\n>>> Seleccionando el modelo ganador (Equilibrio Rendimiento/Estabilidad)...")
        
        # Calculamos medias y desviaciones
        means = df_results.groupby("Classifier")["val_auc"].mean()
        stds = df_results.groupby("Classifier")["val_auc"].std()
        
        # Creamos un score: Penalizamos la desviación restándola de la media
        # Así buscamos el valor más alto de esta combinación
        quality_score = means - stds
        best_model = quality_score.idxmax() 
        
        print(f"GANADOR SELECCIONADO: {best_model}")
        print(f"  --> Score de Calidad (Mean - Std): {quality_score[best_model]:.3f}")
        print(f"  --> Rendimiento (Mean AUC): {means[best_model]:.3f}")
        print(f"  --> Estabilidad (Std Dev): {stds[best_model]:.3f}")
        # ----

        # Name mapping para compatibilidad con los argumentos del Script 3
        model_mapping = {
            "SVM": "SVM",
            "Logistic Regression": "LogisticRegression",
            "Random Forest": "RandomForest",
            "Naive Bayes": "NaiveBayes",
            "KNN": "KNN",
            "Gradient Boosting": "GradientBoosting"
        }
        best_model_finetune = model_mapping.get(best_model, best_model)

        print(f"Fine-tuning best model: {best_model_finetune}")
        path_features = args.csv
        # Construcción del comando para el Script 3
        # Pasamos variables_txt_path para que use exactamente las mismas columnas que el entrenamiento
        fine_tune_cmd = [
            "python3", os.path.join(script_dir, "3_retrain_best_model_and_evaluate_binary.py"),
            "--csv", path_features,                  
            "--model", best_model_finetune,     
            "--variables", variables_txt_path   
        ]

        subprocess.call(fine_tune_cmd)

    print(f"\n>>> [FIN DEL RUN] Todos los procesos han finalizado exitosamente.")
    print(f">>> Directorio de resultados: {experiment_dir}")