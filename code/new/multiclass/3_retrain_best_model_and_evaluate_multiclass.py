import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from copy import deepcopy
from pathlib import Path
import joblib
from sklearn.metrics import make_scorer, roc_auc_score


import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from shap.maskers import Independent

import matplotlib as mpl

import os
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import matplotlib as mpl
mpl.use('Agg')
import scienceplots
plt.style.use(['science', 'grid'])
mpl.rcParams["text.usetex"] = False
dpi = 300

# Libraries for interpretability
import shap
from lime.lime_tabular import LimeTabularExplainer

from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Classifier imports
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Tools for evaluation and calibration
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, f1_score,
    accuracy_score, recall_score, precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# Bayesian optimization for hyperparameters
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import joblib

# For statistical analysis of SHAP values
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import make_scorer, roc_auc_score
from scipy.stats import kruskal
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve





def perform_shap_analysis(X_data, y_data, model_clf, preprocessor, shap_dir, report_path, dataset_name="dataset"):
    """
    Performs SHAP analysis on a dataset.
    
    Args:
        X_data: Raw feature data
        y_data: Labels
        model_clf: Final classifier
        preprocessor: Preprocessing pipeline
        shap_dir: Directory to save results
        dataset_name: Dataset name (for labeling)
    """
    print(f"\nPerforming SHAP analysis for {dataset_name}...")
    try:
        # Apply StandardScaler preserving column names
        scaler = preprocessor.steps[0][1]
        X_scaled = pd.DataFrame(scaler.transform(X_data),
                            index=X_data.index,
                            columns=X_data.columns)
        
        # Apply VarianceThreshold and retrieve selected columns
        vt = preprocessor.steps[1][1]
        mask = vt.get_support()
        selected_features = X_data.columns[mask]
        X_transformed_array = vt.transform(X_scaled.values)
        X_transformed = pd.DataFrame(X_transformed_array,
                                    index=X_data.index,
                                    columns=selected_features)
        
        # # Select appropriate explainer based on model type
        # if isinstance(model_clf, (RandomForestClassifier, GradientBoostingClassifier)):
        #     # For tree-based models
        #     explainer = shap.TreeExplainer(model_clf)
        # elif isinstance(model_clf, LogisticRegression):
        #     # For linear models
        #     try:
        #         explainer = shap.LinearExplainer(model_clf, X_transformed)
        #     except Exception:
        #         # If it fails, use KernelExplainer as alternative
        #         background = shap.kmeans(X_transformed, 50)
        #         explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        # else:
        #     # For other models (SVM, KNN, NaiveBayes)
        #     background = shap.kmeans(X_transformed, 50) # Dataset summary to speed up
        #     explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        
        # test
        classes = np.unique(y_data)
        class_names = {i: f"Class {c}" for i,c in enumerate(classes)}
        print(f" - Clases detectadas: {class_names}")
        shap_cache = os.path.join(shap_dir, "shap_background_cache.pkl")

        if os.path.exists(shap_cache):
            # load previously‐computed explainer result
            shap_result = joblib.load(shap_cache)
            X_slice = X_transformed #shap_result.data
        else:
            # pick a small background to speed things up
            background = X_transformed.sample(min(200, len(X_transformed)), random_state=42)

            # build the explainer on predict_proba
            explainer = shap.Explainer(
                model_clf.predict_proba,      
                masker=Independent(background),
                feature_names=selected_features,
                output_names=class_names       
            )

            # compute SHAP values (on a slice or on the whole)
            X_slice = X_transformed 
            shap_result = explainer(X_slice)    

            # cache it
            joblib.dump(shap_result, shap_cache)

        # now you can immediately extract and plot:
        all_vals = shap_result.values
        print(f" - Valores SHAP calculados para {dataset_name}.")
        n_classes = all_vals.shape[2]
        shap_values_class = [all_vals[:, :, i] for i in range(n_classes)]

        shap.summary_plot(
            shap_values_class,
            X_slice,           # or X_transformed for full
            feature_names=selected_features,
            class_names=class_names,
            class_inds="original",
            show=False
        )
        #adapt the figure size
        plt.gcf().set_size_inches(15, 8)
        plt.savefig(os.path.join(shap_dir, "shap_summary_multiclass.png"), dpi=300)
        plt.savefig(os.path.join(shap_dir, "shap_summary_multiclass.pdf"), dpi=300)

        plt.close()
        
        # --------------------------------------------------------------
        # PART 1: STATISTICAL TEST between SHAP values and class
        # --------------------------------------------------------------
        print(f" - Performing statistical test (Kruskal-Wallis) for {dataset_name} with Holm correction...")

        # Detect unique classes
        unique_classes = np.unique(y_data)
        # List to save results by class
        shap_stats_results = []

        # Process one SHAP matrix per class
        for class_idx, class_shap_values in enumerate(shap_values_class):
            print(f"  > Processing SHAP for class {class_idx}...")

            shap_matrix = pd.DataFrame(
                class_shap_values,
                index=X_transformed.index,
                columns=X_transformed.columns
            )

            features_test = []
            pvalues_raw = []

            for feat in shap_matrix.columns:
                # Group SHAP values of this feature by class
                grouped_values = [shap_matrix.loc[y_data == c, feat] for c in unique_classes]
                try:
                    stat, pval = kruskal(*grouped_values)
                except ValueError:
                    pval = 1.0  # fallback in case one class has no samples
                features_test.append(feat)
                pvalues_raw.append(pval)

            # Holm correction for multiple comparisons
            alpha = 0.05
            reject, pvals_corr, _, _ = multipletests(pvalues_raw, alpha=alpha, method='holm')

            # Prepare the report
            lines_output = []
            lines_output.append("=================================")
            lines_output.append(f"Kruskal-Wallis by feature - SHAP class {class_idx}")
            lines_output.append(f"alpha = {alpha}")
            lines_output.append(f"Total features: {len(features_test)}") 
            lines_output.append("=================================\n")
            lines_output.append(f"Results by feature (raw and corrected p-value):")

            significant_feats = []

            for feat, pval_raw, pval_corr, rej_bool in zip(features_test, pvalues_raw, pvals_corr, reject):
                if rej_bool:
                    result_str = "=> SIGNIFICANT DIFFERENCE"
                    significant_feats.append((feat, pval_raw, pval_corr))
                else:
                    result_str = "=> no significant difference"
                lines_output.append(
                    f"    {feat}: raw p-value={pval_raw:.4e}, corrected p-value={pval_corr:.4e} {result_str}"
                )

            lines_output.append("")
            lines_output.append(f" Total with significant difference: {len(significant_feats)}")

            shap_stats_results.append((class_idx, lines_output))

            # Save result for this class
            test_txt_path = os.path.join(shap_dir, f"shap_statistical_test_class_{class_idx}.txt")
            with open(test_txt_path, "w", encoding="utf-8") as f_out:
                for line in lines_output:
                    f_out.write(line + "\n")

            print(f"    --> Saved: {test_txt_path}")

        # PARTE 2: HEATMAPS - UNO POR CADA CLASE
        # --------------------------------------------------------------
        print(f" - Generando Heatmaps individuales para cada clase en {dataset_name}...")
        
        # Obtener índices ordenados por clase
        class_labels = np.unique(y_data)
        idx_order = np.concatenate([np.where(y_data == c)[0] for c in class_labels])
        class_positions = np.cumsum([len(np.where(y_data == c)[0]) for c in class_labels])
        
        # Crear un heatmap para cada clase
        for i, class_shap in enumerate(shap_values_class):
            print(f"   - Generando heatmap para clase {i}...")
            
            heatmap_path = os.path.join(shap_dir, f"shap_heatmap_class_{i}.png")
            heatmap_path2 = os.path.join(shap_dir, f"shap_heatmap_class_{i}.pdf")

            shap_explanation = shap.Explanation(
                values=class_shap,
                data=X_transformed.values,
                feature_names=list(X_transformed.columns)
            )
            
            shap.plots.heatmap(
                shap_explanation,
                instance_order=idx_order,
                show=False
            )
            
            fig = plt.gcf()
            ax = plt.gca()
            
            # Dibujar líneas divisorias entre clases
            for split in class_positions[:-1]:
                ax.axvline(split - 0.5, color='black', linewidth=2, zorder=10)

            # Etiquetas de clase
            n_total = len(idx_order)
            midpoints=[]
            prev = 0
            # for c, split in zip(class_labels, class_positions):
            #     midpoint = (prev + split) / 2 / n_total
            #     ax.text(midpoint, 1.01, f'Class {c}', ha='center', va='bottom', 
            #            transform=ax.transAxes, fontweight='bold')
            #     prev = split
            # Rotate class labels to avoid overlapping
            for c, split in zip(class_labels, class_positions):
                midpoint = (prev + split) / 2 / n_total
                ax.text(midpoint, 1.01, f'Class {c}', ha='center', va='bottom',
                        transform=ax.transAxes, fontweight='bold', rotation=25)
                prev = split

            # Set the title higher above the plot
            plt.title(f'SHAP Heatmap - Class {i} vs All', fontsize=14, pad=45)
            fig.set_size_inches(12, 10)
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.savefig(heatmap_path2, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"    --> Heatmap clase {i} guardado: {heatmap_path}")



        # --------------
        # NEW Beeswarm plot 
        # --------------
        for i, class_shap in enumerate(shap_values_class):
            shap_fig_path = os.path.join(shap_dir, f"shap_beeswarm_class_{i}.png")
            shap_fig_path2 = os.path.join(shap_dir, f"shap_beeswarm_class_{i}.pdf")

            # Create SHAP Explanation object for this class
            shap_explanation = shap.Explanation(
                values=class_shap,
                data=X_transformed.values,
                feature_names=list(X_transformed.columns)
            )

            shap.plots.beeswarm(shap_explanation, max_display=16, show=False)
            fig = plt.gcf()
            fig.set_size_inches(14, 8)
            plt.subplots_adjust(left=0.4, right=0.95)
            plt.tight_layout()
            plt.savefig(shap_fig_path, dpi=dpi, bbox_inches='tight')
            plt.savefig(shap_fig_path2, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"  --> Beeswarm plot saved: {shap_fig_path}")

    
 
        # --------------------------------------------------------------
        # Scatter plots of top features
        # --------------------------------------------------------------
        # Create directory for individual plots
        top_features_by_class = []
        for i, class_shap in enumerate(shap_values_class):
            mean_abs_shap = np.abs(class_shap).mean(axis=0)  # Remove .values here
            top_idx = np.argsort(mean_abs_shap)[-15:]
            top_idx = top_idx[np.argsort(mean_abs_shap[top_idx])[::-1]]
            top_features_shap = X_transformed.columns[top_idx]
            top_features_by_class.append(top_features_shap)
            
            scatter_dir_class = os.path.join(shap_dir, f"scatter_plots_class_{i}")
            os.makedirs(scatter_dir_class, exist_ok=True)

            # Create SHAP Explanation object for this class
            shap_explanation = shap.Explanation(
                values=class_shap,
                data=X_transformed.values,
                feature_names=list(X_transformed.columns)
            )

            for j, feature in enumerate(top_features_shap, start=1):
                scatter_fig_path = os.path.join(scatter_dir_class, f"{j:02d}_{feature}.png")
                scatter_fig_path2 = os.path.join(scatter_dir_class, f"{j:02d}_{feature}.pdf")
                
                # Get feature index
                feature_idx = X_transformed.columns.get_loc(feature)
                
                shap.plots.scatter(shap_explanation[:, feature_idx], color=shap_explanation, show=False)
                fig = plt.gcf()
                fig.set_size_inches(10, 6)
                plt.tight_layout()
                plt.savefig(scatter_fig_path, dpi=dpi, bbox_inches='tight')
                plt.savefig(scatter_fig_path2, bbox_inches="tight", dpi=300)
                plt.close()
            
            print(f"  --> Scatter plots saved for class {i} in: {scatter_dir_class}")

        return True, selected_features, shap_values_class, top_features_shap
    
    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"=== SHAP Analysis ({dataset_name}) ===\n")
            f_out.write(" Could not generate SHAP (unsupported model or error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        print(f"Error in SHAP analysis for {dataset_name}:", e)
        return False, None, None, None

def main():
    """
    Main function for fine-tuning, evaluation and interpretation of the best model.
    """
    # --- CLI argument configuration ---
    parser = argparse.ArgumentParser(
        description="Trains and fine-tunes a model with best performance."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to features CSV.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["SVM", "LogisticRegression", "RandomForest", 
                                 "NaiveBayes", "KNN", "GradientBoosting"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--variables", type=str, required=True,
                        help="Path to variables_usadas.txt file.")
    args = parser.parse_args()

    # --- 1. Path and Directory Configuration ---
    # Derive output directories based on the variables file location
    selected_model = args.model

    base_dir = os.path.dirname(os.path.abspath(args.variables))
    output_parent_dir = os.path.join(base_dir, f"best_results")
    calibration_dir = os.path.join(output_parent_dir, "calibration")
    explicability_dir = os.path.join(output_parent_dir, "explicability")

    train_explicability_dir = os.path.join(explicability_dir, "train")
    test_explicability_dir = os.path.join(explicability_dir, "test")

    # SHAP and LIME subdirectories for train
    train_shap_dir = os.path.join(train_explicability_dir, "SHAP")
    train_lime_dir = os.path.join(train_explicability_dir, "LIME")

    # SHAP and LIME subdirectories for test
    test_shap_dir = os.path.join(test_explicability_dir, "SHAP")
    test_lime_dir = os.path.join(test_explicability_dir, "LIME")

    # Create directories if they don't exist
    os.makedirs(output_parent_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    os.makedirs(explicability_dir, exist_ok=True)
    os.makedirs(train_explicability_dir, exist_ok=True)
    os.makedirs(test_explicability_dir, exist_ok=True)
    os.makedirs(train_shap_dir, exist_ok=True)
    os.makedirs(train_lime_dir, exist_ok=True)
    os.makedirs(test_shap_dir, exist_ok=True)
    os.makedirs(test_lime_dir, exist_ok=True)

    # --- 2. Data Loading & Preprocessing ---
    df = pd.read_csv(args.csv)
    df = df.set_index(df['patient_id'].astype(str))
    print(f"Data loaded. Shape: {df.shape}")

    # Prepare labels and groups
    y = df['label'].apply(lambda x: x - 1).values
    groups = df['patient_id'].values

    
    # Drop metadata and all 'diagnostics' columns automatically
    cols_to_drop = ['id_igtp', 'study_id', 'label', 'mask_type']
    cols_to_drop += [c for c in df.columns if 'diagnostics' in c]
    
    X = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"Features ready for modeling. Shape: {X.shape}")
    
    
    # ----------------------------------------------------------------------
    # 1.1) FILTER USED VARIABLES (variables_usadas.txt)
    # ----------------------------------------------------------------------
    print(f"\nFiltering variables using file: {args.variables}")
    with open(args.variables, "r", encoding="utf-8") as f_vars:
        used_vars = [line.strip() for line in f_vars if line.strip()]
    X = X[used_vars]

    # ----------------------------------------------------------------------
    # 2) SEPARATE HOLD-OUT TEST SET AND TRAINING SET
    # ----------------------------------------------------------------------
    # gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups= groups))

    X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train_full = groups[train_idx]




# ----------------------------------------------------------------------
    # 3) DEFINE PIPELINE AND SEARCH SPACE WITH BAYESIAN OPTIMIZATION
    # ----------------------------------------------------------------------
    number_folds = args.n_folds
    # Definir las clases presentes (0, 1, 2 si restaste 1 a tus etiquetas 1, 2, 3)
    all_labels = np.unique(y) # [0, 1, 2]

    def robust_auc(y_true, y_score):
        # Forzamos las etiquetas para que coincidan con las columnas de predict_proba
        return roc_auc_score(y_true, y_score, multi_class='ovr', average='macro', labels=all_labels)

    # Crear el scorer usando response_method para versiones nuevas de sklearn
    multiclass_auc_scorer = make_scorer(robust_auc, response_method='predict_proba')

    score_group = {
        'roc_auc': multiclass_auc_scorer,
        'f1': 'f1_macro', # Importante: usar macro para multiclase
        'balanced_accuracy': 'balanced_accuracy'
    }
    score_refit_str = 'roc_auc'  # Metric to select the best model
    random_state_value = 42      # Seed for reproducibility
    
    # --- Specific configuration for each model type ---
    if selected_model == 'SVM':
        # Pipeline for SVM: Scaling → Variance filter → SVM
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             SVC(random_state=random_state_value, probability=True))
        # Search space for hyperparameters
        param_grid = {
            'svc__C': Real(1e-4, 1e3, prior='log-uniform'),            # Regularization
            'svc__kernel': Categorical(['linear', 'rbf', 'poly']),     # Kernel type
            'svc__gamma': Real(1e-4, 1e3, prior='log-uniform'),        # Gamma parameter
            'svc__coef0': Real(0, 1)                                   # Independent term (for poly)
        }
        
    elif selected_model == 'LogisticRegression':
        # Pipeline for Logistic Regression
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             LogisticRegression(
                                 class_weight='balanced', 
                                 random_state=random_state_value,
                                 solver='saga',  
                                 max_iter=10000
                             ))
        # Search space
        param_grid = {
            'logisticregression__C': Real(1e-4, 1e3, prior='log-uniform'),  # Regularization
            'logisticregression__penalty': Categorical(['l1', 'l2', 'elasticnet']),  # Regularization type
            'logisticregression__l1_ratio': Real(0.1, 0.9)                  # L1/L2 ratio for elasticnet
        }
        
    elif selected_model == 'RandomForest':
        # Pipeline for Random Forest
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             RandomForestClassifier(n_jobs=-1, 
                                                    class_weight="balanced_subsample", 
                                                    random_state=random_state_value))
        # Search space
        param_grid = {
            'randomforestclassifier__n_estimators': Integer(50, 1024),       # Number of trees
            'randomforestclassifier__max_depth': Integer(1, 10),             # Maximum depth
            'randomforestclassifier__max_features': Categorical(['sqrt', 'log2', None]),  # Features per tree
            'randomforestclassifier__min_samples_split': Integer(2, 20)      # Min samples to split node
        }
        
    elif selected_model == 'NaiveBayes':
        # Pipeline for Naive Bayes
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             GaussianNB())
        param_grid = {'gaussiannb__var_smoothing': Real(1e-11, 1e-7, prior='log-uniform')}  # Naive Bayes has no hyperparameters to optimize (added to estabilization)
        
    elif selected_model == 'KNN':
        # Pipeline for K-Nearest Neighbors
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             KNeighborsClassifier(n_jobs=-1))
        # Search space
        param_grid = {
            'kneighborsclassifier__n_neighbors': Integer(2, 8),            # Number of neighbors
            'kneighborsclassifier__weights': Categorical(['uniform', 'distance'])  # Weighting
        }
        
    elif selected_model == 'GradientBoosting':
        # Pipeline for Gradient Boosting
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             GradientBoostingClassifier(random_state=random_state_value))
        # Search space
        param_grid = {
            'gradientboostingclassifier__n_estimators': Integer(50, 1024),        # Number of trees
            'gradientboostingclassifier__learning_rate': Real(1e-4, 0.1, prior='log-uniform'),  # Learning rate
            'gradientboostingclassifier__max_depth': Integer(1, 10),              # Maximum depth
            'gradientboostingclassifier__subsample': Real(0.5, 1.0),              # Sample fraction per tree
            'gradientboostingclassifier__max_features': Categorical(['sqrt', 'log2', None])  # Features per tree
        }
    else:
        raise ValueError(f"Model '{selected_model}' not recognized.")
    
    # ----------------------------------------------------------------------
    # 4) FIT WITH BayesSearchCV (BAYESIAN OPTIMIZATION) ON TRAINING SET
    # ----------------------------------------------------------------------
    
    cv = StratifiedGroupKFold(n_splits=number_folds, shuffle=True, random_state=random_state_value)
    print("\nStarting Bayesian optimization with BayesSearchCV...")
    estimator_path = os.path.join(output_parent_dir, "best_estimator.pkl")
    search_path = os.path.join(output_parent_dir, "search_results.pkl")
    
    if os.path.exists(estimator_path) and os.path.exists(search_path):
        print(f"\n>>> Pre-trained multiclass model found. Loading...")
        best_estimator = joblib.load(estimator_path)
        search = joblib.load(search_path)
    else:
        print(f"\n>>> No model found. Starting Bayesian Search for {selected_model}...")
        
        # Configure Bayesian search with the specific multiclass scorer
        search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_grid,
            scoring=score_group,
            refit=score_refit_str,
            cv=cv,
            n_jobs=-1,
            random_state=random_state_value
        )
        
        # Train on the hold-out training set
        search.fit(X_train_full, y_train_full, groups=groups_train_full)
        best_estimator = search.best_estimator_
        
        # Save results for future runs
        joblib.dump(best_estimator, estimator_path)
        joblib.dump(search, search_path)
        print(f"  --> Optimization finished. Model saved at: {os.path.relpath(estimator_path)}")

    print(f"\nOptimization complete. Best params: {search.best_params_}")
    

    # ----------------------------------------------------------------------
    # 5) SAVE REPORT IN "report.txt"
    # ----------------------------------------------------------------------

    report_path = os.path.join(output_parent_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"=== Model {selected_model} fine-tuning ===\n\n")
        f_out.write(f"Best parameters (according to {score_refit_str}): {search.best_params_}\n\n")
        f_out.write("=== CV Results (BayesSearch) ===\n")
        idx_best = search.best_index_
        for key in score_group:
            mean_test = search.cv_results_[f'mean_test_{key}'][idx_best]
            std_test  = search.cv_results_[f'std_test_{key}'][idx_best]
            f_out.write(f"  CV {key}: {mean_test:.3f} +/- {std_test:.3f}\n")
        f_out.write("\n")


    # ----------------------------------------------------------------------
    # 6) EVALUATE ON TEST
    # ----------------------------------------------------------------------
    print("\nEvaluating model on test set (uncalibrated)...")
    y_pred_test = best_estimator.predict(X_test)

    # Generate uncalibrated confusion matrix
    # --- Generate and save confusion matrix (Uncalibrated) ---
    confusion_fig_png = os.path.join(output_parent_dir, "confusion_matrix.png")
    confusion_fig_pdf = os.path.join(output_parent_dir, "confusion_matrix.pdf")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.grid(False)

    disp = ConfusionMatrixDisplay.from_estimator(
        best_estimator,
        X_test,
        y_test,
        ax=ax,
        cmap='Blues',
        colorbar=False
    )

    n_classes = len(disp.display_labels)
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Improve axis labels and title
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    plt.tight_layout()
    plt.savefig(confusion_fig_png, dpi=dpi, bbox_inches='tight')
    plt.savefig(confusion_fig_pdf, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Confusion matrix saved at: {confusion_fig_png}")
    confusion_fig = confusion_fig_png  # for report

    # Calculate performance metrics on test (multiclass AUC)
    if hasattr(best_estimator, "predict_proba"):
        y_prob = best_estimator.predict_proba(X_test)
        auc_   = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
    elif hasattr(best_estimator, "decision_function"):
        df = best_estimator.decision_function(X_test)
        auc_ = roc_auc_score(
            y_test,
            df,
            multi_class="ovr",
            average="macro"
        )
    else:
        auc_ = np.nan

    # General classification metrics
    mcc_    = matthews_corrcoef(y_test, y_pred_test)
    kappa_  = cohen_kappa_score(y_test, y_pred_test)
    # f1_     = f1_score(y_test, y_pred_test)
    acc_    = accuracy_score(y_test, y_pred_test)
    balacc_ = balanced_accuracy_score(y_test, y_pred_test)
        # General classification metrics (multiclass)
    f1_     = f1_score(y_test, y_pred_test, average='macro')            # <-- specify average
    recall_macro_ = recall_score(y_test, y_pred_test, average='macro')  # <-- multiclass recall
    precision_macro_ = precision_score(y_test, y_pred_test, average='macro')  # <-- multiclass precision

    # Detailed classification report
    report_cr = classification_report(y_test, y_pred_test)

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Test Evaluation (NOT calibrated) ===\n")
        f_out.write(f"  Confusion Matrix Figure: {confusion_fig}\n")
        f_out.write(f"  AUC: {auc_:.3f}\n")
        f_out.write(f"  MCC: {mcc_:.3f}\n")
        f_out.write(f"  Kappa: {kappa_:.3f}\n")
        f_out.write(f"  F1: {f1_:.3f}\n")
        f_out.write(f"  Accuracy: {acc_:.3f}\n")
        f_out.write(f"  Recall: {recall_macro_:.3f}\n")
        f_out.write(f"  Precision: {precision_macro_:.3f}\n")
        f_out.write(f"  Balanced Accuracy: {balacc_:.3f}\n\n")
        f_out.write("=== Classification Report ===\n")
        f_out.write(report_cr)
        f_out.write("\n\n")

    # --- Calibrate with Platt scaling (sigmoid, cv=5) ---
    print("\nCalibrating model with Platt scaling (sigmoid, cv=5)...")
    cal_clf = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=5)
    cal_clf.fit(X_train_full, y_train_full)

    print(f"  --> Model calibrated successfully")




    # one-hot encode the test labels
    classes    = [1,2,3,4,5]
    y_test_bin = label_binarize(y_test, classes=classes)

    # pre- and post- calibration probabilities
    probas_pre  = best_estimator.predict_proba(X_test)
    probas_post = cal_clf.predict_proba (X_test)

    # Plot
    for tag, probas in [("pre",  probas_pre),
                        ("post", probas_post)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(classes):
            prob_true, prob_pred = calibration_curve(
                y_test_bin[:, i],
                probas[:, i],
                n_bins=10,
                strategy="uniform"
            )
            ax.plot(prob_pred,
                    prob_true,
                    marker="o",
                    label=f"Class {cls}")
        ax.plot([0,1], [0,1], "k:", label="Ideal")
        ax.set_title(f"Calibration curves ({tag})")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed probability")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(calibration_dir,
                                f"calibration_curve_{tag}.png"),
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close(fig)



    # Calibration metrics 
    def calibration_error(y_true, y_prob, n_bins=10, norm='l1'):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_prob = np.asarray(y_prob, dtype=np.float64)

        if norm not in {"l1", "l2"}:
            raise ValueError("norm must be 'l1' or 'l2'")

        # 1. Assign each probability to a bin
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)      

        # 2. Sum weighted error by the size of each bin
        ece = 0.0
        N = len(y_true)
        for i in range(n_bins):
            idx = bin_ids == i
            if idx.any():
                prob_avg = y_prob[idx].mean()
                acc_avg  = y_true[idx].mean()
                err_bin  = abs(prob_avg - acc_avg) if norm == "l1" else (prob_avg - acc_avg) ** 2
                ece     += err_bin * idx.sum() / N

        return ece

    # 1) Probabilities (without / with Platt)
    p_pre  = best_estimator.predict_proba(X_test)      # shape (n_samples, 5)
    p_post = cal_clf.predict_proba(X_test)


    def multiclass_calibration_error(y_true, y_prob, n_bins=10, norm='l1'):
        """
        Compute the multiclass Expected Calibration Error (ECE) by
        averaging the class-wise Brier‐style ECE.
        
        y_true: array-like of shape (n_samples,) with integer labels 0..K-1
        y_prob: array-like of shape (n_samples, K) with class probabilities
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        classes = np.unique(y_true)
        K = len(classes)
        
        # one-hot encode
        y_true_bin = label_binarize(y_true, classes=classes)  # shape (n, K)
        
        ece_total = 0.0
        N = len(y_true)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        
        # for each class, compute its ECE
        for k in range(K):
            pk = y_prob[:, k]
            tk = y_true_bin[:, k]
            
            # assign each prediction to a bin
            bin_ids = np.digitize(pk, bins, right=True) - 1
            bin_ids = np.clip(bin_ids, 0, n_bins - 1)
            
            ece_k = 0.0
            for i in range(n_bins):
                mask = bin_ids == i
                if not mask.any():
                    continue
                prob_avg = pk[mask].mean()
                acc_avg  = tk[mask].mean()
                err = abs(prob_avg - acc_avg) if norm == 'l1' else (prob_avg - acc_avg)**2
                ece_k += err * mask.sum() / N
            
            ece_total += ece_k
        
        return ece_total / K



    # NEW — multiclass
    ece_pre  = multiclass_calibration_error(y_test, p_pre,  n_bins=10, norm='l1')
    ece_post = multiclass_calibration_error(y_test, p_post, n_bins=10, norm='l1')
    # 3) Brier score
    brier_pre  = np.mean(np.sum((p_pre  - y_test_bin) ** 2, axis=1))
    brier_post = np.mean(np.sum((p_post - y_test_bin) ** 2, axis=1))

    # 4) Dump results to report
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Calibration Metrics ===\n")
        f_out.write(f"  ECE  (pre):  {ece_pre:.3f}\n")
        f_out.write(f"  ECE  (post): {ece_post:.3f}\n")
        f_out.write(f"  Brier (pre): {brier_pre:.3f}\n")
        f_out.write(f"  Brier (post): {brier_post:.3f}\n\n")

    # --- Threshold adjustment to optimize F1 ---
    thresholds = np.linspace(0.1, 0.9, 9)
    best_thresh = None
    best_f1 = -np.inf
    results = []

    # Find optimal threshold for F1
    for thresh in thresholds:
        y_pred_thresh = (cal_clf.predict_proba(X_test)[:, 1] >= thresh).astype(int)
        f1_val = f1_score(y_test, y_pred_thresh,average='macro')
        results.append({'threshold': thresh, 'f1': f1_val})
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh

    # Generate predictions with optimal threshold
    y_pred_best = (cal_clf.predict_proba(X_test)[:, 1] >= best_thresh).astype(int)


    probs_post = cal_clf.predict_proba(X_test)
    auc_best        = roc_auc_score(y_test, probs_post, multi_class='ovr', average='macro')
    mcc_best        = matthews_corrcoef(y_test, y_pred_best)
    kappa_best      = cohen_kappa_score(y_test, y_pred_best)
    f1_best_macro   = f1_score(y_test, y_pred_best, average='macro')
    precision_best  = precision_score(y_test, y_pred_best, average='macro')
    recall_best     = recall_score(y_test, y_pred_best, average='macro')
    acc_best        = accuracy_score(y_test, y_pred_best)
    balacc_best     = balanced_accuracy_score(y_test, y_pred_best)

    report_cr_best = classification_report(y_test, y_pred_best)


    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Threshold Adjustment (Results with best threshold) ===\n")
        f_out.write("Results for each threshold:\n")
        for r in results:
            f_out.write(f"  Threshold: {r['threshold']:.2f} – F1 (macro): {r['f1']:.3f}\n")
        f_out.write(f"\nBest threshold selected (according to F1 macro): {best_thresh:.2f}\n\n")
        f_out.write(f"Classification Report (threshold={best_thresh:.2f}):\n")
        f_out.write(report_cr_best)
        f_out.write("\n")
        f_out.write(f"ROC-AUC (ovr, macro): {auc_best:.3f}\n")
        f_out.write(f"MCC: {mcc_best:.3f}\n")
        f_out.write(f"Kappa: {kappa_best:.3f}\n")
        f_out.write(f"F1 (macro): {f1_best_macro:.3f}\n")
        f_out.write(f"Precision (macro): {precision_best:.3f}\n")
        f_out.write(f"Recall (macro): {recall_best:.3f}\n")
        f_out.write(f"Accuracy: {acc_best:.3f}\n")
        f_out.write(f"Balanced Accuracy: {balacc_best:.3f}\n\n")


    # --- Calibrated confusion matrix ---
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    confusion_fig_best = os.path.join(calibration_dir, "confusion_matrix_best_threshold.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)

    disp_best = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best)
    disp_best.plot(ax=ax, cmap='cividis')
    # ax.set_title(f"{selected_model} (Calibrado, threshold={best_thresh:.2f})", fontsize=12)

    n_classes = conf_matrix_best.shape[0]

    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)

    ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(confusion_fig_best, dpi=dpi, bbox_inches='tight')
    plt.close()

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write(f"Confusion Matrix (Calibrated with threshold={best_thresh:.2f}) fig: {confusion_fig_best}\n\n")
        

    # ----------------------------------------------------------------------
    # 7) EXPLAINABILITY
    # ----------------------------------------------------------------------

    # Extract the preprocessor (all steps except the final classifier)
    preprocessor = deepcopy(best_estimator)
    preprocessor.steps.pop(-1)

    # Extract the final classifier
    model_clf = best_estimator.steps[-1][1]

    # Perform SHAP analysis for training set
    train_success, selected_features, train_shap_values, train_top_features = perform_shap_analysis(
        X_data=X_train_full,
        y_data=y_train_full,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=train_shap_dir,
        report_path=report_path,
        dataset_name="training"
    )

    # Perform SHAP analysis for test set
    test_success, _, test_shap_values, test_top_features = perform_shap_analysis(
        X_data=X_test,
        y_data=y_test,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=test_shap_dir,
        report_path=report_path,
        dataset_name="test"
    )

  
            
    print(f"\nProcess finished. Report saved at: {report_path}")

if __name__ == "__main__":
    main()