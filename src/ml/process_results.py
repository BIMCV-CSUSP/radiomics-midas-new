import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("test_multiple_models_mode.csv")


# Function to get best result for a given group
def get_best_result(group):
    return group.loc[group["mean_test_score"].idxmax()]


# Overall best result
best_overall = get_best_result(df)
print("Best overall result:")
print(
    best_overall[
        ["Model", "Image Type", "Experiment", "mean_test_score", "param_reduce_dim"]
    ]
)
print("\n")

# Best result per model
best_per_model = df.groupby("Model").apply(get_best_result)
print("Best result per model:")
print(
    best_per_model[["Image Type", "Experiment", "mean_test_score", "param_reduce_dim"]]
)
print("\n")

# Differences between image types
image_type_comparison = (
    df.groupby(["Model", "Image Type", "Experiment"])["mean_test_score"]
    .max()
    .unstack(level="Image Type")
)
print("Best scores for each model, experiment, and image type:")
print(image_type_comparison)
print("\n")

print("Difference between best and worst image type for each model and experiment:")
print(image_type_comparison.max(axis=1) - image_type_comparison.min(axis=1))
print("\n")


# Differences between feature selection strategies
def get_reduction_method(param):
    if pd.isna(param):
        return "No Reduction"
    elif "CorrelationFeatureReduction" in str(param):
        return "CorrelationFeatureReduction"
    elif "mRMRFeatureReduction" in str(param):
        return "mRMRFeatureReduction"
    elif "PCA" in str(param):
        return "PCA"
    else:
        return "Other"


df["reduction_method"] = df["param_reduce_dim"].apply(get_reduction_method)

reduction_comparison = (
    df.groupby(["Model", "Experiment", "reduction_method"])["mean_test_score"]
    .max()
    .unstack(level="reduction_method")
)
print("Best scores for each model, experiment, and reduction method:")
print(reduction_comparison)
print("\n")

print(
    "Difference between best and worst reduction method for each model and experiment:"
)
print(reduction_comparison.max(axis=1) - reduction_comparison.min(axis=1))
print("\n")

# Best result per experiment
best_per_experiment = df.groupby("Experiment").apply(get_best_result)
print("Best result per experiment:")
print(
    best_per_experiment[
        [
            "Model",
            "Image Type",
            "mean_test_score",
            "param_reduce_dim",
            "param_reduce_dim__K",
            "param_reduce_dim__n_components",
        ]
    ]
)
print("\n")

# Comparison across experiments
experiment_comparison = (
    df.groupby(["Model", "Experiment"])["mean_test_score"].max().unstack()
)
print("Best scores for each model and experiment:")
print(experiment_comparison)
print("\n")

print("Difference between best and worst experiment for each model:")
print(experiment_comparison.max(axis=1) - experiment_comparison.min(axis=1))
