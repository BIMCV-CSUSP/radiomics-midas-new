#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for optimization and advanced evaluation of the best model.

This script performs:
1. Hyperparameter fine-tuning through Bayesian search
2. Evaluation on a hold-out test set
3. Model probability calibration
4. Interpretability analysis with SHAP and LIME
"""
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

# Bayesian optimization of hyperparameters
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import joblib

# For statistical analysis of SHAP values
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def save_plot_both_formats(fig_path_base, dpi=300, bbox_inches='tight'):
    """
    Helper function to save plots in both PNG and PDF formats.
    
    Args:
        fig_path_base: Base path without extension (e.g., '/path/to/figure')
        dpi: DPI for PNG format
        bbox_inches: bbox_inches parameter for plt.savefig
    """
    # Save as PNG
    png_path = f"{fig_path_base}.png"
    plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    
    # Save as PDF
    pdf_path = f"{fig_path_base}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches=bbox_inches)
    
    print(f"  --> Figure saved as PNG: {png_path}")
    print(f"  --> Figure saved as PDF: {pdf_path}")

###########################
#         SHAP            #
###########################

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
        
        # Apply VarianceThreshold and recover selected columns
        vt = preprocessor.steps[1][1]
        mask = vt.get_support()
        selected_features = X_data.columns[mask]
        X_transformed_array = vt.transform(X_scaled.values)
        X_transformed = pd.DataFrame(X_transformed_array,
                                    index=X_data.index,
                                    columns=selected_features)
        # Create directory for SHAP results
        os.makedirs(shap_dir, exist_ok=True)
        shap_cache = os.path.join(shap_dir, "shap_values.pkl")

        if os.path.exists(shap_cache):
            # load previously‐computed explainer result
            shap_values = joblib.load(shap_cache)
        else:
            # Select appropriate explainer based on model type
            if isinstance(model_clf, (RandomForestClassifier, GradientBoostingClassifier)):
                # For tree-based models
                explainer = shap.TreeExplainer(model_clf)
            elif isinstance(model_clf, LogisticRegression):
                # For linear models
                try:
                    explainer = shap.LinearExplainer(model_clf, X_transformed)
                except Exception:
                    # If it fails, use KernelExplainer as alternative
                    background = shap.kmeans(X_transformed, 50)
                    explainer = shap.KernelExplainer(model_clf.predict_proba, background)
            else:
                # For other models (SVM, KNN, NaiveBayes)
                background = shap.kmeans(X_transformed, 50) # Dataset summary to speed up
                explainer = shap.KernelExplainer(model_clf.predict_proba, background)
            
            # Calculate SHAP values
            shap_values = explainer(X_transformed)
            joblib.dump(shap_values, os.path.join(shap_dir, 'shap_values.pkl'))
        
        # For binary classification, keep SHAP values for positive class
        if shap_values.values.ndim > 2:
            shap_values = shap_values[:,:,1]
        
        # --------------------------------------------------------------
        # PART 1: STATISTICAL TEST between SHAP values and class
        # --------------------------------------------------------------
        print(f" - Performing statistical test (Mann-Whitney U) for {dataset_name} with Holm correction...")
        
        # Build DataFrame with SHAP values
        shap_matrix = pd.DataFrame(
            shap_values.values,
            index=X_transformed.index,
            columns=X_transformed.columns
        )
        
        # Lists for statistical results
        features_test = []
        pvalues_raw = []
        
        # Perform test for each feature
        for feat in shap_matrix.columns:
            # Separate SHAP values by class
            shap_class0 = shap_matrix.loc[y_data == 0, feat]
            shap_class1 = shap_matrix.loc[y_data == 1, feat]
            
            # Mann-Whitney U test (non-parametric test to compare distributions)
            stat, pval = mannwhitneyu(shap_class0, shap_class1, alternative='two-sided')
            features_test.append(feat)
            pvalues_raw.append(pval)
        
        # Multiple comparisons correction (Holm method)
        alpha = 0.05
        reject, pvals_corr, _, _ = multipletests(pvalues_raw, alpha=alpha, method='holm')
        
        # Generate results report
        lines_output = []
        lines_output.append("=================================")
        lines_output.append(f"MANN-WHITNEY U TEST (SHAP by feature) with 'Holm' correction for {dataset_name}")
        lines_output.append("Comparison: Class 0 vs Class 1")
        lines_output.append(f"alpha = {alpha}")
        lines_output.append(f"Total features: {len(features_test)}") 
        lines_output.append("=================================\n")
        
        lines_output.append(f"Results by feature (raw and corrected p-value):")
        significant_feats = []
        
        # Process results for each feature
        for feat, pval_raw, pval_corr, rej_bool in zip(features_test, pvalues_raw, pvals_corr, reject):
            if rej_bool: # Significant difference (reject H0)
                result_str = "=> SIGNIFICANT DIFFERENCE"
                significant_feats.append((feat, pval_raw, pval_corr))
            else:
                result_str = "=> no significant difference"
            
            lines_output.append(
                f"    {feat}: raw p-value={pval_raw:.4e}, corrected p-value={pval_corr:.4e} {result_str}"
            )
        # Summary of significant features
        lines_output.append("")
        lines_output.append(f" Total comparisons with significant difference: {len(significant_feats)}. Comparisons:")
        
        if not significant_feats:
            lines_output.append("    No significant differences found.")
        else:
            for feat, pval_raw, pval_corr in significant_feats:
                lines_output.append(
                    f"    {feat}: raw p-value={pval_raw:.4e}, corrected p-value={pval_corr:.4e} => SIGNIFICANT DIFFERENCE"
                )
        
        lines_output.append("\n")
        
        # Save results to file
        test_txt_path = os.path.join(shap_dir, "shap_statistical_test.txt")
        with open(test_txt_path, "w", encoding="utf-8") as f_out:
            for line in lines_output:
                f_out.write(line + "\n")
        
        print(f"  --> Statistical test saved at: {test_txt_path}")
    
        # --------------------------------------------------------------
        # PART 2: HEATMAP (class 0 first, then class 1)
        # --------------------------------------------------------------
        print(f" - Generating Heatmap with samples ordered by class for {dataset_name}...")
        # Order instances by class for visualization
        idx_class0 = np.where(y_data == 0)[0]
        idx_class1 = np.where(y_data == 1)[0]
        
        # Concatenate indices (first class 0, then class 1)
        idx_order = np.concatenate([idx_class0, idx_class1])
        
        heatmap_path = os.path.join(shap_dir, "shap_heatmap.png")
        
        # Generate heatmap with SHAP, specifying instance order
        shap.plots.heatmap(
            shap_values, 
            show=False,
            instance_order=idx_order
        )
        
        fig = plt.gcf()
        ax = plt.gca()
        
        # Add dividing line between classes
        split_position = len(idx_class0)
        ax.axvline(split_position - 0.5, color='black', linewidth=1, zorder=10)
        n_total = len(idx_order)
        mid_class0 = (split_position / 2) / n_total
        mid_class1 = (split_position + len(idx_class1)/2) / n_total
        
        # Add labels above the heatmap
        ax.text(mid_class0, 1.01, 'Class 0', ha='center', va='bottom', transform=ax.transAxes)
        ax.text(mid_class1, 1.01, 'Class 1', ha='center', va='bottom', transform=ax.transAxes)
        
        # Save figure
        fig.set_size_inches(10, 6)
        plt.tight_layout()
        heatmap_path_base = os.path.splitext(heatmap_path)[0]  # Remove .png extension
        save_plot_both_formats(heatmap_path_base, dpi=300)
        plt.close()
        print(f"  --> Reordered heatmap saved at: {heatmap_path_base}.png and {heatmap_path_base}.pdf")
    
        # --------------
        # Beeswarm plot 
        # --------------
        shap_fig_path = os.path.join(shap_dir, "shap_beeswarm.png")
        shap.plots.beeswarm(shap_values, max_display=16, show=False)
        fig = plt.gcf()
        fig.set_size_inches(14, 8)
        plt.subplots_adjust(left=0.4, right=0.95)
        plt.tight_layout()
        shap_fig_path_base = os.path.splitext(shap_fig_path)[0]  # Remove .png extension
        save_plot_both_formats(shap_fig_path_base, dpi=dpi)
        plt.close()
        print(f"  --> Beeswarm plot saved at: {shap_fig_path_base}.png and {shap_fig_path_base}.pdf")
    
        # --------------------------------------------------------------
        # Scatter plots for top features
        # --------------------------------------------------------------
        # Create directory for individual plots
        scatter_dir = os.path.join(shap_dir, "scatter_plots")
        os.makedirs(scatter_dir, exist_ok=True)
        # Identify the 15 features with highest impact (absolute SHAP value)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-15:]
        top_idx = top_idx[np.argsort(mean_abs_shap[top_idx])[::-1]]
        top_features_shap = X_transformed.columns[top_idx]
        
        for i, feature in enumerate(top_features_shap, start=1):
            scatter_fig_path = os.path.join(scatter_dir, f"{i:02d}_{feature}.png")
            shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            plt.tight_layout()
            scatter_fig_path_base = os.path.splitext(scatter_fig_path)[0]  # Remove .png extension
            save_plot_both_formats(scatter_fig_path_base, dpi=dpi)
            plt.close()
        
        print(f"  --> Scatter plots for most relevant variables saved at: {scatter_dir}")
        return True, selected_features, shap_values, top_features_shap
    
    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"=== SHAP Analysis ({dataset_name}) ===\n")
            f_out.write(" Could not generate SHAP (model not supported or error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        print(f"Error in SHAP analysis for {dataset_name}:", e)
        return False, None, None, None



def main():
    """
    Main function for fine-tuning, evaluation and interpretation of the best model.
    """
    # --- Initial configuration and command line arguments ---
    parser = argparse.ArgumentParser(
        description="Trains and fine-tunes a model using a definitive hold-out test set and cross-validation on the remaining data. Then calibrates the model and applies SHAP if possible."
    )
    parser.add_argument("--csv", type=str, default="features_all_gland.csv",
                        help="Path to CSV with features (default 'features_all_gland.csv').")
    parser.add_argument("--model", type=str, required=True,
                        choices=["SVM", "LogisticRegression", "RandomForest", 
                                 "NaiveBayes", "KNN", "GradientBoosting"],
                        help="Model to train/optimize.")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for cross-validation in BayesSearchCV")
    parser.add_argument("--variables", type=str, default="../../../results/radiomics/most_discriminant/gland/variables_usadas.txt",
                        help="Path to variables_usadas.txt file with variables to use.")
    args = parser.parse_args()
    
    print("\nStarting model fine-tuning.")
    print(f"  --> Selected model: {args.model}")
    print(f"  --> CSV used: {args.csv}")
    print(f"  --> Variables file: {args.variables}")

    # Configuration of paths and output directories
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
    
    print(f"\nOutput folder created/located at: {os.path.relpath(output_parent_dir)}")
    
    # ----------------------------------------------------------------------
    # 1) LOAD CSV AND IDENTIFY X, y, groups
    # ----------------------------------------------------------------------
    pre_path = "../../../artifacts/radiomics"
    data_filename = str(args.csv) if args.csv else "features_all_gland.csv"
    data_path = os.path.join(pre_path, "concatenated_data", data_filename)
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    df['patient_id_type'] = df['patient_id'].astype(str)
    df = df.set_index('patient_id_type')
    print(f"Data loaded. Dimensions: {df.shape}")
    
    # Prepare variables for modeling
    y = df['label'].values
    groups = df['patient_id'].values
    X = df.drop(columns=['patient_id'])
    
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
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train_full = groups[train_idx]
    
    # ----------------------------------------------------------------------
    # 3) DEFINE PIPELINE AND SEARCH SPACE WITH BAYESIAN OPTIMIZATION
    # ----------------------------------------------------------------------
    number_folds = args.n_folds
    score_group = {
        'roc_auc': 'roc_auc',
        'f1': 'f1',
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
        param_grid = {}  # Naive Bayes has no hyperparameters to optimize
        
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
    
    # Definir rutas de persistencia basadas en el directorio de salida dinámico
    estimator_path = os.path.join(output_parent_dir, "best_estimator.pkl")
    search_path = os.path.join(output_parent_dir, "search_results.pkl")

    if os.path.exists(estimator_path) and os.path.exists(search_path):
        print(f"\n>>> Modelo pre-entrenado encontrado en {output_parent_dir}. Cargando...")
        best_estimator = joblib.load(estimator_path)
        search = joblib.load(search_path)
    else:
        print("\n>>> No se encontró modelo previo. Iniciando optimización bayesiana...")
        search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_grid,
            scoring=score_group,
            refit=score_refit_str,
            cv=cv,
            n_jobs=-1,
            random_state=random_state_value
        )
        search.fit(X_train_full, y_train_full, groups=groups_train_full)
        best_estimator = search.best_estimator_
        
        # Guardar para futuras ejecuciones
        joblib.dump(best_estimator, estimator_path)
        joblib.dump(search, search_path)
        print(f"  --> Modelo y resultados de búsqueda guardados en: {output_parent_dir}")

    print(f"\nOptimization completed. Best parameters: {search.best_params_}")



    
    # ----------------------------------------------------------------------
    # 5) SAVE REPORT IN "report.txt"
    # ----------------------------------------------------------------------
    report_path = os.path.join(output_parent_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"=== Fine-tuning of {selected_model} model ===\n\n")
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
    print("\n[Evaluation] Uncalibrated test set performance...")
    y_pred_test = best_estimator.predict(X_test)

    # Paths for confusion matrix (PNG + PDF)
    confusion_fig  = os.path.join(output_parent_dir, "confusion_matrix.png")
    confusion_fig2 = os.path.join(output_parent_dir, "confusion_matrix.pdf")

    # Confusion matrix (uncalibrated)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    disp = ConfusionMatrixDisplay.from_estimator(
        estimator=best_estimator,
        X=X_test,
        y=y_test,
        ax=ax,
        cmap="Blues",
        colorbar=False
    )
    n_classes = len(disp.display_labels)
    
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(confusion_fig, dpi=dpi, bbox_inches='tight')
    plt.savefig(confusion_fig2, dpi=dpi, bbox_inches='tight')
    print(f"Confusion matrix saved at: {confusion_fig} and {confusion_fig2}")
    plt.close()
    print(f"Confusion matrix saved at: {confusion_fig}")
    
    # Calculate performance metrics on test
    if hasattr(best_estimator, "predict_proba"):
        auc_ = roc_auc_score(y_test, best_estimator.predict_proba(X_test)[:, 1])
    elif hasattr(best_estimator, "decision_function"):
        auc_ = roc_auc_score(y_test, best_estimator.decision_function(X_test))
    else:
        auc_ = np.nan
    
    # General classification metrics
    mcc_    = matthews_corrcoef(y_test, y_pred_test)
    kappa_  = cohen_kappa_score(y_test, y_pred_test)
    f1_     = f1_score(y_test, y_pred_test)
    acc_    = accuracy_score(y_test, y_pred_test)
    sens_   = recall_score(y_test, y_pred_test, pos_label=1)
    spec_   = recall_score(y_test, y_pred_test, pos_label=0)
    ppv_    = precision_score(y_test, y_pred_test, pos_label=1)
    npv_    = precision_score(y_test, y_pred_test, pos_label=0)
    balacc_ = balanced_accuracy_score(y_test, y_pred_test)
    
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
        f_out.write(f"  Sensitivity: {sens_:.3f}\n")
        f_out.write(f"  Specificity: {spec_:.3f}\n")
        f_out.write(f"  PPV: {ppv_:.3f}\n")
        f_out.write(f"  NPV: {npv_:.3f}\n")
        f_out.write(f"  Balanced Accuracy: {balacc_:.3f}\n\n")
        f_out.write("=== Classification Report ===\n")
        f_out.write(report_cr)
        f_out.write("\n\n")
    
    # --- Calibrate with Platt scaling (sigmoid, cv=5) ---
    print("\nCalibrating model with Platt scaling (sigmoid, cv=5)...")
    cal_clf = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=5)
    cal_clf.fit(X_train_full, y_train_full)
    
    # --- Calibration curve PRE (before calibrating) ---
    calibration_fig_pre = os.path.join(calibration_dir, "calibration_pre.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_estimator(
        best_estimator, 
        X_test, 
        y_test, 
        n_bins=10, 
        name=f"{selected_model}_pre", 
        ax=ax
    )

    for line in ax.get_lines():
        line.set_color("black")

    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color("black")
        for line in legend.get_lines():
            line.set_color("black")
        for patch in legend.get_patches():
            patch.set_edgecolor("black")
            patch.set_facecolor("black")
            
    # ax.set_title(f"Calibration Curve (pre), {selected_model}", fontsize=14)
    plt.savefig(calibration_fig_pre, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  --> Calibration curve (pre) saved at: {calibration_fig_pre}")
    
    # --- Calibration curve POST (after calibrating) ---
    calibration_fig_post = os.path.join(calibration_dir, "calibration_post.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_estimator(
        cal_clf, 
        X_test, 
        y_test, 
        n_bins=10, 
        name=f"{selected_model}_post", 
        ax=ax
    )
    # ax.set_title(f"Calibration Curve (post), {selected_model}", fontsize=14)

    for line in ax.get_lines():
        line.set_color("black")

    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color("black")
        for line in legend.get_lines():
            line.set_color("black")
        for patch in legend.get_patches():
            patch.set_edgecolor("black")
            patch.set_facecolor("black")
            
    plt.savefig(calibration_fig_post, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  --> Calibration curve (post) saved at: {calibration_fig_post}")

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

        # 2. Sum weighted error by bin size
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
    p_pre  = best_estimator.predict_proba(X_test)[:, 1]
    p_post = cal_clf.predict_proba(X_test)[:, 1]

    # 2) Expected Calibration Error (ECE)
    ece_pre  = calibration_error(y_test, p_pre,  n_bins=10, norm='l1')
    ece_post = calibration_error(y_test, p_post, n_bins=10, norm='l1')

    # 3) Brier score
    brier_pre  = brier_score_loss(y_test, p_pre)
    brier_post = brier_score_loss(y_test, p_post)

    # 4) Write results to report
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Calibration metrics ===\n")
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
        f1_val = f1_score(y_test, y_pred_thresh)
        results.append({'threshold': thresh, 'f1': f1_val})
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh
    
    # Generate predictions with optimal threshold
    y_pred_best = (cal_clf.predict_proba(X_test)[:, 1] >= best_thresh).astype(int)
    
    # Calculate metrics with optimized threshold
    auc_best    = roc_auc_score(y_test, cal_clf.predict_proba(X_test)[:, 1])
    mcc_best    = matthews_corrcoef(y_test, y_pred_best)
    kappa_best  = cohen_kappa_score(y_test, y_pred_best)
    f1_best     = f1_score(y_test, y_pred_best)
    acc_best    = accuracy_score(y_test, y_pred_best)
    sens_best   = recall_score(y_test, y_pred_best, pos_label=1)
    spec_best   = recall_score(y_test, y_pred_best, pos_label=0)
    ppv_best    = precision_score(y_test, y_pred_best, pos_label=1)
    npv_best    = precision_score(y_test, y_pred_best, pos_label=0)
    balacc_best = balanced_accuracy_score(y_test, y_pred_best)
    
    report_cr_best = classification_report(y_test, y_pred_best)

    # Save calibration and threshold adjustment results
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Threshold Adjustment (Results with best threshold) ===\n")
        f_out.write("Results for each threshold:\n")
        for r in results:
            f_out.write("Threshold: {:.2f} - F1: {:.3f}\n".format(r['threshold'], r['f1']))
        f_out.write(f"\nBest threshold selected (according to F1): {best_thresh:.2f}\n")
        f_out.write("\nClassification Report (with threshold {:.2f}):\n".format(best_thresh))
        f_out.write(report_cr_best)
        f_out.write("\n")
        f_out.write(f"AUC: {auc_best:.3f}\n")
        f_out.write(f"MCC: {mcc_best:.3f}\n")
        f_out.write(f"Kappa: {kappa_best:.3f}\n")
        f_out.write(f"F1: {f1_best:.3f}\n")
        f_out.write(f"Accuracy: {acc_best:.3f}\n")
        f_out.write(f"Sensitivity: {sens_best:.3f}\n")
        f_out.write(f"Specificity: {spec_best:.3f}\n")
        f_out.write(f"PPV: {ppv_best:.3f}\n")
        f_out.write(f"NPV: {npv_best:.3f}\n")
        f_out.write(f"Balanced Accuracy: {balacc_best:.3f}\n\n")
    
    # --- Calibrated confusion matrix ---
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    confusion_fig_best = os.path.join(calibration_dir, "confusion_matrix_best_threshold.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    
    disp_best = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best)
    disp_best.plot(ax=ax, cmap='cividis')
    # ax.set_title(f"{selected_model} (Calibrated, threshold={best_thresh:.2f})", fontsize=12)
    
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
    # 7) INTERPRETABILITY
    # ----------------------------------------------------------------------

    # Extract the preprocessor (all steps except final classifier)
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

    print(f"\nProcess completed. Report saved at: {report_path}")

if __name__ == "__main__":
    main()