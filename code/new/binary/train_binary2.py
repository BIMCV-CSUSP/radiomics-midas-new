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
    df = pd.read_csv(args.csv)
    y = df["label"].values
    groups = df["patient_id"].values
    # # Identificar columnas a eliminar (Leakage Prevention)
    cols_to_drop = ['id_igtp','patient_id', 'study_id', 'label', 'mask_type','SSA_type']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    # # Eliminar columnas de diagnóstico (metadatos de radiómica)
    X = X.drop(columns=[c for c in X.columns if 'diagnostics' in c], errors='ignore')


    from scipy.stats import shapiro, ttest_ind, mannwhitneyu
    from sklearn import metrics

    feature_names = []
    pvalues = []
    aucs = []
    sensitivities = []
    specificities = []
    cutoffs = []
    tests_used = []

    for col in X.columns:
        x_col = X[col]

        _, p_norm = shapiro(x_col)
        a = x_col[y == 0]
        b = x_col[y == 1]

        if p_norm > 0.05:
            _, pval = ttest_ind(a, b)
            test = "t-test"
        else:
            _, pval = mannwhitneyu(a, b)
            test = "mann-whitney"

        fpr, tpr, thresholds = metrics.roc_curve(y, x_col, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)

        if auc_val < 0.5:
            fpr, tpr, thresholds = metrics.roc_curve(y, x_col, pos_label=0)
            auc_val = metrics.auc(fpr, tpr)

        youden = tpr - fpr
        best_idx = np.argmax(youden)

        feature_names.append(col)
        pvalues.append(pval)
        aucs.append(auc_val)
        sensitivities.append(tpr[best_idx])
        specificities.append(1 - fpr[best_idx])
        cutoffs.append(thresholds[best_idx])
        tests_used.append(test)

    train_auc_pvals_df = pd.DataFrame({
        "AUC": aucs,
        "Cutoff": cutoffs,
        "Sensitivity": sensitivities,
        "Specificity": specificities,
        "Test": tests_used,
        "p-value": pvalues
    }, index=feature_names).sort_values(by="p-value")

    os.makedirs(fs_dir, exist_ok=True)
    train_auc_pvals_df.to_csv(os.path.join(fs_dir, "ranking_univariante_global.csv"))

    print(">> Ranking univariante global generado")

        

    # ==============================================================================
    # Bloque 2: Selección de Características (Estrategia LASSO Multivariante)
    # ==============================================================================
    
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFECV

    def select_features_train_only(
        X_train,
        y_train,
        corr_threshold=0.85,
        min_features=2
    ):
        """
        Selección de características TRAIN-ONLY (anti-leakage).

        Pipeline:
        1) Ranking univariante (screening)
        2) Eliminación de redundancia (Spearman)
        3) Selección multivariante (RFECV + Logistic L1)
        """

        # ---------- 1. Ranking univariante (TRAIN) ----------
        pvals = {}
        for col in X_train.columns:
            _, p_norm = shapiro(X_train[col])
            a = X_train[col][y_train == 0]
            b = X_train[col][y_train == 1]
            _, pval = ttest_ind(a, b) if p_norm > 0.05 else mannwhitneyu(a, b)
            pvals[col] = pval

        ranked_cols = sorted(pvals, key=pvals.get)
        X_ranked = X_train[ranked_cols]

        # ---------- 2. Filtro de correlación ----------
        corr = X_ranked.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [
            col for col in upper.columns
            if any(upper[col] > corr_threshold)
        ]

        X_clean = X_ranked.drop(columns=to_drop)

        # ---------- 3. RFECV multivariante ----------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        base_model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            class_weight='balanced',
            C=1.0,
            max_iter=5000
        )

        selector = RFECV(
            estimator=base_model,
            scoring='roc_auc',
            cv=5,
            step=1,
            min_features_to_select=min_features
        )

        selector.fit(X_scaled, y_train)

        selected_features = X_clean.columns[selector.support_].tolist()

        # ---------- Cláusula de rescate ----------
        if len(selected_features) < min_features:
            selected_features = ranked_cols[:min_features]

        return selected_features

    # ==============================================================================
    # Bloque 3: Entrenamiento y Evaluación de Modelos (Cross-Validation)  
    # ==============================================================================
    
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score, confusion_matrix
    import ast

    results_filepath = os.path.join(experiment_dir, "results.csv")
    preds_filepath   = os.path.join(experiment_dir, "predictions.csv")
    features_filepath = os.path.join(experiment_dir, "features_per_fold.csv")

    # ------------------------------------------------------------------
    # REANUDAR SI EXISTE
    # ------------------------------------------------------------------

    if os.path.exists(results_filepath) and os.path.exists(preds_filepath):
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

    else:
        print("\n>>> Iniciando entrenamiento con CV y selección por fold...")

        models = get_models(random_state=42)

        all_results = []
        all_predictions = []
        all_selected_features = []

        cv = StratifiedGroupKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=42
        )

        global_fold = 0

        for model_name, model in models:
            print(f"\nEvaluating {model_name}...")

            for fold, (train_idx, val_idx) in enumerate(
                cv.split(X, y, groups), start=1
            ):
                global_fold += 1

                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # -----------------------------------
                # SELECCIÓN DE VARIABLES (TRAIN ONLY)
                # -----------------------------------
                selected_features = select_features_train_only(X_train, y_train)

                X_train_sel = X_train[selected_features]
                X_val_sel   = X_val[selected_features]

                # Guardar variables usadas en este fold
                all_selected_features.append({
                    "Classifier": model_name,
                    "Fold": global_fold,
                    "Selected_Features": selected_features
                })

                # ---------------------------
                # ENTRENAMIENTO
                # ---------------------------
                model.fit(X_train_sel, y_train)

                # ---------------------------
                # PREDICCIÓN
                # ---------------------------
                y_val_pred = model.predict(X_val_sel)

                if hasattr(model, "predict_proba"):
                    y_val_prob = model.predict_proba(X_val_sel)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_val_prob = model.decision_function(X_val_sel)
                else:
                    y_val_prob = None

                # ---------------------------
                # MÉTRICAS
                # ---------------------------
                auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else np.nan
                cm = confusion_matrix(y_val, y_val_pred)

                sens = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else np.nan
                spec = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else np.nan

                all_results.append({
                    "Classifier": model_name,
                    "Fold": global_fold,
                    "val_auc": auc,
                    "val_sensitivity": sens,
                    "val_specificity": spec,
                    "n_features": len(selected_features)
                })

                all_predictions.append({
                    "Classifier": model_name,
                    "Fold": global_fold,
                    "y_val": y_val.tolist(),
                    "y_pred": y_val_pred.tolist(),
                    "y_prob": y_val_prob.tolist() if y_val_prob is not None else []
                })

        # ------------------------------------------------------------------
        # GUARDAR RESULTADOS
        # ------------------------------------------------------------------

        df_results = pd.DataFrame(all_results)
        df_results.to_csv(results_filepath, index=False)

        df_preds = pd.DataFrame(all_predictions)
        df_preds.to_csv(preds_filepath, index=False)

        df_features = pd.DataFrame(all_selected_features)
        df_features.to_csv(features_filepath, index=False)

        print(f"\nResultados guardados en: {results_filepath}")
        print(f"Predicciones guardadas en: {preds_filepath}")
        print(f"Variables por fold guardadas en: {features_filepath}")



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


    best_results_dir = os.path.join(experiment_dir, "best_results")
    features_per_fold_path = os.path.join(experiment_dir, "features_per_fold.csv")

    if os.path.exists(best_results_dir):
        print(f"\n>>> El re-entrenamiento del mejor modelo ya existe en {best_results_dir}. Saltando...")
    else:
        print("\n>>> Seleccionando el modelo ganador (equilibrio rendimiento / estabilidad)...")

        # ----------------------------------------------------------------------
        # 1) Selección del mejor modelo
        # ----------------------------------------------------------------------
        means = df_results.groupby("Classifier")["val_auc"].mean()
        stds = df_results.groupby("Classifier")["val_auc"].std()

        # Score de calidad = rendimiento medio - penalización por inestabilidad
        quality_score = means - stds
        best_model = quality_score.idxmax()

        print(f"GANADOR SELECCIONADO: {best_model}")
        print(f"  --> Score de calidad (Mean - Std): {quality_score[best_model]:.3f}")
        print(f"  --> Mean AUC: {means[best_model]:.3f}")
        print(f"  --> Std AUC: {stds[best_model]:.3f}")

        # ----------------------------------------------------------------------
        # 2) Mapeo de nombre a formato esperado por Script 3
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # 3) Construcción del comando para Script 3 corregido
        # ----------------------------------------------------------------------
        fine_tune_cmd = [
            "python3",
            os.path.join(script_dir, "3_retrain_best_model_and_evaluate_binary.py"),
            "--csv", args.csv,
            "--model", best_model_finetune,
            "--outdir", best_results_dir,
            "--n_folds", str(args.n_splits)
        ]

        # Si existe el archivo con variables seleccionadas por fold, se pasa
        # para derivar una firma estable en el Script 3
        if os.path.exists(features_per_fold_path):
            fine_tune_cmd.extend([
                "--features_per_fold", features_per_fold_path
            ])

        # ----------------------------------------------------------------------
        # 4) Ejecución robusta del Script 3
        # ----------------------------------------------------------------------
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