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
parser = argparse.ArgumentParser(
    description="Model evaluation with repeated cross-validation"
)
parser.add_argument(
    "--csv", type=str,
    choices=["features_all_gland.csv", "features_all_full.csv"],
    default="features_all_gland.csv",
    help="Name of CSV file with features."
)
parser.add_argument(
    "--data_pre", type=str,
    default="../../../artifacts/radiomics",
    help="Root directory where radiomics data is located."
)
parser.add_argument(
    "--results_base", type=str, default="../../../results/radiomics",
    help="Base directory where results will be created."
)
parser.add_argument(
    "--n_splits", type=int, default=5,
    help="Number of partitions for StratifiedGroupKFold (per repetition)."
)
parser.add_argument(
    "--n_repeats", type=int, default=10,
    help="Number of cross-validation repetitions."
)
parser.add_argument(
    "--feature_strategy", type=str,
    choices=["all", "most_discriminant"],
    default="most_discriminant",
    help="Feature selection strategy: 'all' or 'most_discriminant'."
)
parser.add_argument(
    "--calculate_differences", action="store_true", default=True,
    help="If enabled, executes model_differences.py."
)
parser.add_argument(
    "--fine_tune_best_model", action="store_true", default=False,
    help="If enabled, performs fine-tuning of the best model."
)

args = parser.parse_args(args=[])


# # --- Data loading and preprocessing ---
# label_csv= "label5" 
# num_label = label_csv[-1]
path_features = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/binary/features_t2w_BPfirrmann.csv"
df = pd.read_csv(path_features)
y = df["label"]
groups = df["patient_id"]
X = df.drop([ 'patient_id','study_id','label', 'mask_type',
                              'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 
                              'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 
                              'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings', 
                              'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash', 
                              'diagnostics_Image-original_Dimensionality', 'diagnostics_Image-original_Spacing', 
                              'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean', 
                              'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 
                              'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing', 
                              'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox', 
                              'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 
                              'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass', 
                              'diagnostics_Image-interpolated_Spacing', 'diagnostics_Image-interpolated_Size', 
                              'diagnostics_Image-interpolated_Mean', 'diagnostics_Image-interpolated_Minimum', 
                              'diagnostics_Image-interpolated_Maximum', 'diagnostics_Mask-interpolated_Spacing', 
                              'diagnostics_Mask-interpolated_Size', 'diagnostics_Mask-interpolated_BoundingBox', 
                              'diagnostics_Mask-interpolated_VoxelNum', 'diagnostics_Mask-interpolated_VolumeNum', 
                              'diagnostics_Mask-interpolated_CenterOfMassIndex', 'diagnostics_Mask-interpolated_CenterOfMass', 
                              'diagnostics_Mask-interpolated_Mean', 'diagnostics_Mask-interpolated_Minimum', 
                              'diagnostics_Mask-interpolated_Maximum'], axis=1)

experiment_dir = "/mnt/datalake/openmind/MedP-Midas/sgonzalez/radiomics-midas-new/binary"
os.makedirs(experiment_dir, exist_ok=True)


# --- Feature selection ---
selected_features = X.columns

if args.feature_strategy == "most_discriminant":
    print(">> Performing feature selection...")

    # Directories for feature selection results
    fs_dir = os.path.join(experiment_dir, "feature_selection")
    os.makedirs(fs_dir, exist_ok=True)
    images_dir = os.path.join(fs_dir, f"images")
    os.makedirs(images_dir, exist_ok=True)

# Initialize lists to store statistics per feature
    feature_names, sensitivity_list, specificity_list = ([] for _ in range(3))
    auc_list, threshold_list, test_type_list, pvalue_list, pos_vs_neg_list = ([] for _ in range(5))
    
    # Evaluate each feature individually
    for column in X.columns:
        # Shapiro-Wilk normality test
        stat, p = shapiro(X[column])
        
        # Get distributions by class
        a_dist = X[column][y == 0]  # Class 0(1-3)
        b_dist = X[column][y == 1]  # Class 1(4-5)
        
        feature_names.append(column)
        
        # Select statistical test according to normality
        alpha = 0.05
        if p > alpha: # If p > 0.05, assume normality
            test_type_list.append('t-test')
            _, pval = ttest_ind(a_dist, b_dist) # T-test for normal data
        else:
            test_type_list.append('mann-whitney U-test')
            _, pval = mannwhitneyu(a_dist, b_dist) # Non-parametric test
        pvalue_list.append(pval)
        
        # Evaluate discriminative capacity (AUC)
        fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=1)
        auc_val = metrics.auc(fpr, tpr)

        # If AUC < 0.5, invert the relationship (greater/lesser)
        pos_vs_neg = ">" 
        if auc_val < 0.5:
            fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=0)
            auc_val = metrics.auc(fpr, tpr)
            pos_vs_neg = "<"
        auc_list.append(auc_val)
        pos_vs_neg_list.append(pos_vs_neg)
        
        # Find optimal point in ROC curve (Youden's J index)
        roc_df = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            '1-fpr': 1 - fpr,
            'tf': tpr - (1 - fpr),   # Youden's J = Sensitivity + Specificity - 1
            'thresholds': thresholds
        })
        cutoff_df = roc_df.iloc[(roc_df.tf - 0).abs().argsort()[:1]] # Closest point to optimal
        
        # Save sensitivity, specificity and optimal threshold
        sensitivity_list.append(cutoff_df['tpr'].values[0])
        specificity_list.append(cutoff_df['1-fpr'].values[0])
        threshold_list.append(cutoff_df['thresholds'].values[0])
    
    # Create DataFrame with all statistics per feature
    train_auc_pvals_df = pd.DataFrame(
        list(zip(auc_list, pos_vs_neg_list, threshold_list,
                    sensitivity_list, specificity_list, 
                    test_type_list, pvalue_list)),
        index=feature_names,
        columns=['AUC', 'Pos.vs.Neg.', 'Cutoff-Threshold', 'Sensitivity',
                    'Specificity', 'Test', 'p-value']
    ).sort_values(by='p-value', ascending=True) #

    # Select features: maximum 1 feature per 15 samples
    num_features_model = round(X.shape[0] / 15)
    train_df = train_auc_pvals_df.sort_values(by='p-value', ascending=True)

    # Select the N most significant features
    selected_features = train_df.index[0:num_features_model]
    print(f"  --> Selected {len(selected_features)} most relevant features.")

    # Filter DataFrame to use only selected features
    X = X[selected_features]
    # Save DataFrame with complete statistics
    df_path_1 = os.path.join(fs_dir, f"train_auc_pvals_df.csv")
    train_auc_pvals_df.loc[selected_features].to_csv(df_path_1)
    print(f"  --> Saved CSV: {df_path_1}\n")

    top_20 = train_auc_pvals_df.index[:20]


    for rank, feature_name in enumerate(top_20, start=1):
        # Create filename
        safe_feat_name = feature_name.replace("/", "_")
        feat_folder_name = f"{rank}_{safe_feat_name}"
        feat_folder_path = os.path.join(images_dir, feat_folder_name)
        os.mkdir(feat_folder_path)
        
        # 1. Violin plot to visualize distributions by class
        plt.figure(figsize=(9, 9))
        sns.violinplot(x=y, y=df[feature_name], color='grey')
        plt.title(f"Distribution of {feature_name} in 0 vs 1", fontsize=14)
        plt.xlabel("Classes")
        plt.xticks([0, 1], ["0", "1"], fontsize=12)
        violin_plot_path = os.path.join(feat_folder_path, f"{safe_feat_name}_violinplot.png")
        plt.savefig(violin_plot_path, dpi=dpi)
        plt.close()
        
        # 2. ROC curve for this individual feature
        fpr, tpr, _ = metrics.roc_curve(y, df[feature_name], pos_label=1)
        auc_val = metrics.auc(fpr, tpr)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, marker='.', color='black', markersize=3, label=f"{feature_name} (AUC={auc_val:.3f})")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title(f"ROC Curve: Discriminative capacity of {feature_name}")
        roc_plot_path = os.path.join(feat_folder_path, f"{safe_feat_name}_ROC.png")
        plt.savefig(roc_plot_path, dpi=dpi)
        plt.close()
else:
    print(">> Using ALL features (without selection).")
    
#correlation matrix of the features in english only with the first 20 features
features_for_corr = X.columns[:200]
corr_matrix = X[features_for_corr].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("Features Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
corr_plot_path = os.path.join(experiment_dir, "correlation_matrix.png")
plt.savefig(corr_plot_path, dpi=dpi)
print(f"  --> Saved correlation matrix: {corr_plot_path}\n")
    
    
# # --- Model training and evaluation ---
# models = get_models(random_state=42)

# # Collectors for results
# all_results = []
# preds_data = []

# # Evaluate each model
# for model_name, model in models:
#     print(f"Evaluating {model_name}...")
#     fold_metrics_list, pred_vals = evaluate_model(
#         model, X, y, groups,
#         n_splits=args.n_splits,
#         n_repeats=args.n_repeats,
#         base_random_state=42
#     )

#     # Add classifier name to each result
#     for fold_metrics in fold_metrics_list:
#         fold_metrics["Classifier"] = model_name
#         all_results.append(fold_metrics)

#     # Store predictions
#     preds_data.append({
#         "Classifier": model_name,
#         "folds": pred_vals["folds"]
#     })

# # Create DataFrame with all results
# df_results = pd.DataFrame(all_results)

# # Sort columns for better readability
# fixed_cols = ["Classifier", "Fold", "Repeat"]
# other_cols = [c for c in df_results.columns if c not in fixed_cols]
# df_results = df_results[fixed_cols + other_cols]
# df_results.sort_values(by=["Classifier", "Fold"], inplace=True)

# # Generate filename for results
# results_filename = f"results_lumbar_discs_label.csv"

# # Save results
# results_filepath = os.path.join(experiment_dir, results_filename)
# df_results.to_csv(results_filepath, index=False)
# print(f"\nResults saved at '{results_filepath}'")



# # --- Structure prediction data for saving ---
# records_for_csv = []
# for item in preds_data:
#     clf_name = item["Classifier"]
#     folds_info = item["folds"]
#     for fold_info in folds_info:
#         fold_idx = fold_info["fold_index"]
#         rep_idx = fold_info["Repeat"]
        
#         y_val_list = fold_info["y_val"].tolist()
#         y_pred_list = fold_info["y_val_pred"].tolist()
#         if fold_info["y_val_prob"] is not None:
#             y_prob_list = fold_info["y_val_prob"].tolist()
#         else:
#             y_prob_list = []
        
#         records_for_csv.append({
#             "Classifier": clf_name,
#             "Fold": fold_idx,
#             "Repeat": rep_idx,
#             "y_val": y_val_list,
#             "y_pred": y_pred_list,
#             "y_prob": y_prob_list
#         })

# # Save predictions to CSV
# df_preds = pd.DataFrame(records_for_csv)
# preds_filename = f"preds_lumbar_discs_label.csv"
# preds_filepath = os.path.join(experiment_dir, preds_filename)
# df_preds.to_csv(preds_filepath, index=False)
# print(f"Predictions saved at '{preds_filepath}'")


# --- Ejecución de análisis adicionales (scripts complementarios) ---
results_filename = f"results_lumbar_discs_label.csv"
results_filepath = os.path.join(experiment_dir, results_filename)
df_results = pd.read_csv(results_filepath)

preds_filename = f"preds_lumbar_discs_label.csv"
preds_filepath = os.path.join(experiment_dir, preds_filename)
df_preds = pd.read_csv(preds_filepath)


# --- Save list of used variables ---
variables_txt_path = os.path.join(experiment_dir, f"variables_used.txt")
with open(variables_txt_path, "w") as f:
    for feat in selected_features:
        f.write(str(feat) + "\n")
print(f"File with used variables: {variables_txt_path}")


# # --- ROC curves generation ---
# print("\nGenerating ROC curves: optimal and median fold per classifier...")

# roc_dir = os.path.join(experiment_dir, f"ROC_curves")
# os.makedirs(roc_dir, exist_ok=True)

# # Collectors for ROC curve information
# curves_info_optimal = []  # For fold with best AUC of each model
# curves_info_median = []   # For fold with median AUC of each model

# # Process each classifier
# classifiers = df_results["Classifier"].unique()
# for clf_name in classifiers:
#     df_clf = df_results[df_results["Classifier"] == clf_name]
    
#     # --- Identify optimal fold (best AUC) ---
#     best_fold_idx = df_clf["val_auc"].idxmax()
#     best_fold_num = df_clf.loc[best_fold_idx, "Fold"]
    
#     # --- Identify median fold (AUC closest to median) ---
#     median_auc = df_clf["val_auc"].median()
#     median_fold_idx = (df_clf["val_auc"] - median_auc).abs().idxmin()
#     median_fold_num = df_clf.loc[median_fold_idx, "Fold"]
    
#     # --- Process data for optimal fold ---
#     df_clf_preds_best = df_preds[
#         (df_preds["Classifier"] == clf_name) & 
#         (df_preds["Fold"] == best_fold_num)
#     ]
#     if len(df_clf_preds_best) > 0:
#         y_val_list_best = df_clf_preds_best.iloc[0]["y_val"]
#         y_prob_list_best = df_clf_preds_best.iloc[0]["y_prob"]
#         if y_prob_list_best:
#             fpr_best, tpr_best, _ = metrics.roc_curve(y_val_list_best, y_prob_list_best, pos_label=1)
#             auc_val_best = metrics.auc(fpr_best, tpr_best)
#             curves_info_optimal.append({
#                 "classifier": clf_name,
#                 "fold": best_fold_num,
#                 "fpr": fpr_best,
#                 "tpr": tpr_best,
#                 "auc": auc_val_best
#             })
    
#     # --- Process data for median fold ---
#     df_clf_preds_median = df_preds[
#         (df_preds["Classifier"] == clf_name) & 
#         (df_preds["Fold"] == median_fold_num)
#     ]
#     if len(df_clf_preds_median) > 0:
#         y_val_list_median = df_clf_preds_median.iloc[0]["y_val"]
#         y_prob_list_median = df_clf_preds_median.iloc[0]["y_prob"]
#         if y_prob_list_median:
#             fpr_median, tpr_median, _ = metrics.roc_curve(y_val_list_median, y_prob_list_median, pos_label=1)
#             auc_val_median = metrics.auc(fpr_median, tpr_median)
#             curves_info_median.append({
#                 "classifier": clf_name,
#                 "fold": median_fold_num,
#                 "fpr": fpr_median,
#                 "tpr": tpr_median,
#                 "auc": auc_val_median
#             })

# # Sort curves of each type by descending AUC (best performance first)
# curves_info_optimal.sort(key=lambda x: x["auc"], reverse=True)
# curves_info_median.sort(key=lambda x: x["auc"], reverse=True)

# # Define color palette for visual consistency between plots
# my_colors = [
#     "#0072B2",  # Dark blue
#     "#009E73",  # Green
#     "#D55E00",  # Reddish orange
#     "#CC78BC",  # Purple
#     "#DE8F05",  # Brown/orange
#     "#56B4E9"   # Light blue
# ]

# my_palette = sns.color_palette(my_colors)

# # Assign a specific color to each classifier
# fixed_classifiers = ["SVM", "Logistic Regression", "Random Forest", 
#                         "Naive Bayes", "KNN", "Gradient Boosting"]
# color_mapping = {clf: my_palette[i] for i, clf in enumerate(fixed_classifiers)}

# # --- Generate ROC plot for optimal folds ---
# fig_opt, ax_opt = plt.subplots(figsize=(8, 6))
# for info in curves_info_optimal:
#     clf_name = info["classifier"]
#     fold_num = info["fold"]
#     fpr = info["fpr"]
#     tpr = info["tpr"]
#     auc_val = info["auc"]
#     ax_opt.plot(fpr, tpr, label=f"{clf_name} (Fold={fold_num}, AUC={auc_val:.3f})", 
#                 color=color_mapping[clf_name])

# # Add diagonal reference line (random classification)
# ax_opt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="_nolegend_")
# ax_opt.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
# ax_opt.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
# ax_opt.tick_params(axis='both', which='major', labelsize=10)
# ax_opt.legend(fontsize=10)

# # Improve legend appearance
# leg = ax_opt.get_legend()
# for line in leg.get_lines():
#     line.set_linewidth(2.5)
# fig_opt.tight_layout()

# # Save plot
# roc_plot_path_opt = os.path.join(roc_dir, "roc_optimal_folds.png")
# plt.savefig(roc_plot_path_opt, dpi=dpi, bbox_inches='tight')
# plt.close(fig_opt)
# print(f"ROC plot (optimal fold) saved at: {roc_plot_path_opt}")

# # --- Generate ROC plot for median folds ---
# fig_med, ax_med = plt.subplots(figsize=(8, 6))
# for info in curves_info_median:
#     clf_name = info["classifier"]
#     fold_num = info["fold"]
#     fpr = info["fpr"]
#     tpr = info["tpr"]
#     auc_val = info["auc"]
#     ax_med.plot(fpr, tpr, label=f"{clf_name} (Fold={fold_num}, AUC={auc_val:.3f})", 
#                 color=color_mapping[clf_name])
    
# # Add diagonal reference line
# ax_med.plot([0, 1], [0, 1], linestyle='--', color='gray', label="_nolegend_")
# ax_med.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
# ax_med.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
# ax_med.tick_params(axis='both', which='major', labelsize=10)
# ax_med.legend(fontsize=10)

# # Improve legend appearance
# leg = ax_med.get_legend()
# for line in leg.get_lines():
#     line.set_linewidth(2.5)
# fig_med.tight_layout()

# # Save plot
# roc_plot_path_med = os.path.join(roc_dir, "roc_median_folds.png")
# plt.savefig(roc_plot_path_med, dpi=dpi, bbox_inches='tight')
# plt.close(fig_med)
# print(f"ROC plot (median fold) saved at: {roc_plot_path_med}")



# =====================================================================

# --- Additional analysis execution (complementary scripts) ---

# Execute statistical model comparison script (optional)

# print("\nExecuting model comparisons (model_differences.py)...")
# import subprocess

# model_diff_dir = os.path.join(experiment_dir, f"model_differences")
# os.mkdir(model_diff_dir)

# # Build command for comparison script
# postprocess_cmd = [
#     "python3",
#     "2_model_differences.py",
#     "--csv_preds", preds_filepath,  # File with predictions
#     "--csv_results", results_filepath,  # File with metrics
#     "--metric", "val_auc",  # Metric to compare (AUC)
#     "--alpha", "0.05",  # Significance level
#     "--outdir", model_diff_dir  # Output directory
# ]

# # Execute script as separate process
# subprocess.call(postprocess_cmd)



# # Fine-tuning of best model (optional)
# # Identify the best model (first in sorted list)
# best_model = "Gradient Boosting"  # Default value
# # best_model = curves_info_optimal[0]["classifier"]


# # Name mapping for fine-tuning script
# model_mapping = {
#     "SVM": "SVM",
#     "Logistic Regression": "LogisticRegression",
#     "Random Forest": "RandomForest",
#     "Naive Bayes": "NaiveBayes",
#     "KNN": "KNN",
#     "Gradient Boosting": "GradientBoosting"
# }
# best_model_finetune = model_mapping.get(best_model, best_model)

# print(f"Fine-tuning best model: {best_model_finetune}")

# # Build command for fine-tuning script
# fine_tune_cmd = [
#     "python3",
#     "3_retrain_best_model_and_evaluate.py",
#     "--csv", path_features,                  # Same feature CSV
#     "--model", best_model_finetune,     # Best identified model
#     "--variables", variables_txt_path   # Selected variables
# ]

# subprocess.call(fine_tune_cmd)

    
