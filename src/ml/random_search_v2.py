from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from transforms import mRMRFeatureReduction


def random_search_cv(experiment, clf, search_grid, features, labels):
    # Create a stratified 10-fold cross-validation object
    skf = StratifiedKFold(n_splits=10)

    # Perform cross-validation
    pipeline_clf = Pipeline(
        [
            ("reduce_dim", "passthrough"),
            ("classifier", clf),
        ]
    )

    rs_clf = RandomizedSearchCV(
        pipeline_clf,
        search_grid,
        cv=skf,
        scoring="f1_weighted",
        n_iter=100,
        n_jobs=-1,
        random_state=0,
        verbose=2,
    )
    search_results = rs_clf.fit(features, labels)

    results = pd.Series(
        {
            "Model": clf.__class__.__qualname__,
            "F1": search_results.best_score_,
            "Params": {
                key.replace("classifier__", ""): value
                for key, value in search_results.best_params_.items()
                if key.startswith("classifier__")
            },
        }
    )
    results.name = experiment
    return results


if __name__ == "__main__":
    import argparse

    np.random.seed(0)

    parser = argparse.ArgumentParser(
        prog="Random search",
        description="This program performs a random search over a pre-defined parameter grid"
        "for the best models for each task (output of `test_multiple_models.py`)",
    )
    parser.add_argument(
        "features",
        type=Path,
        help="the path to the radiomics features CSV",
    )
    parser.add_argument(
        "labels",
        type=Path,
        help="the path to the labels CSV",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="the path to the output CSV",
    )
    args = parser.parse_args()
    features_path = args.features
    labels_path = args.labels
    output = args.output

    linear_svm_distribution = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", "auto"] + list(np.logspace(-3, 3, 7)),
    }

    gradient_boosting_distribution = {
        "classifier__n_estimators": [50, 100, 200, 500],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__max_features": ["auto", "sqrt", "log2"],
    }

    logistic_regression_distribution = {
        "classifier__C": np.logspace(-3, 3, 7),
        "classifier__penalty": ["l1", "l2", "elasticnet"],
        "classifier__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "classifier__max_iter": [100, 200, 300],
    }

    mlp_distribution = {
        "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "classifier__activation": ["relu", "tanh"],
        "classifier__solver": ["lbfgs", "adam", "sgd"],
        "classifier__alpha": [0.0001, 0.001, 0.01],
        "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
    }

    experiments = {
        "5 levels/L5-S": {
            "classifier": LogisticRegression(),
            "distribution": {
                "reduce_dim": [PCA(n_components=0.95, random_state=0)],
                **logistic_regression_distribution,
            },
            "disc": 1,
        },
        "5 levels/L4-L5": {
            "classifier": GradientBoostingClassifier(),
            "distribution": gradient_boosting_distribution,
            "disc": 2,
        },
        "5 levels/L3-L4": {
            "classifier": MLPClassifier(),
            "distribution": {
                "reduce_dim": [mRMRFeatureReduction(K=20)],
                **mlp_distribution,
            },
            "disc": 3,
        },
        "5 levels/L2-L3": {
            "classifier": GradientBoostingClassifier(),
            "distribution": {
                "reduce_dim": [mRMRFeatureReduction(K=20)],
                **gradient_boosting_distribution,
            },
            "disc": 4,
        },
        "5 levels/L1-L2": {
            "classifier": MLPClassifier(),
            "distribution": {
                "reduce_dim": [mRMRFeatureReduction(K=20)],
                **mlp_distribution,
            },
            "disc": 5,
        },
        "5 levels/ALL": {
            "classifier": LogisticRegression(),
            "distribution": logistic_regression_distribution,
        },
        "4 levels/L5-S": {
            "classifier": SVC(kernel="linear"),
            "distribution": {
                "reduce_dim": [PCA(n_components=0.95, random_state=0)],
                **linear_svm_distribution,
            },
            "disc": 1,
        },
        "4 levels/L4-L5": {
            "classifier": GradientBoostingClassifier(),
            "distribution": gradient_boosting_distribution,
            "disc": 2,
        },
        "4 levels/L3-L4": {
            "classifier": MLPClassifier(),
            "distribution": {
                "reduce_dim": [mRMRFeatureReduction(K=20)],
                **mlp_distribution,
            },
            "disc": 3,
        },
        "4 levels/L2-L3": {
            "classifier": GradientBoostingClassifier(),
            "distribution": {
                "reduce_dim": [mRMRFeatureReduction(K=10)],
                **gradient_boosting_distribution,
            },
            "disc": 4,
        },
        "4 levels/L1-L2": {
            "classifier": MLPClassifier(),
            "distribution": {
                "reduce_dim": [PCA(n_components=0.99, random_state=0)],
                **mlp_distribution,
            },
            "disc": 5,
        },
        "4 levels/ALL": {
            "classifier": SVC(kernel="linear"),
            "distribution": linear_svm_distribution,
        },
    }

    results = []

    for key, config in experiments.items():
        labels = pd.read_csv(labels_path, index_col="ID")
        features = pd.read_csv(features_path, index_col="ID")
        if "ALL" not in key:
            labels = labels.loc[labels.index.str.endswith(str(config["disc"]))]
            features = features.loc[features.index.str.endswith(str(config["disc"]))]
        if "4" in key:
            labels[labels == 1] = 2
        classifier = config["classifier"]
        result = random_search_cv(
            key, classifier, config["distribution"], features, labels
        )
        print(result)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output)
