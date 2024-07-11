from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from transforms import mRMRFeatureReduction


def compute_metrics(
    experiment, clf, params, train_features, train_labels, test_features, test_labels
):
    # Create a stratified 10-fold cross-validation object
    skf = StratifiedKFold(n_splits=10)

    # Perform cross-validation
    pipeline_clf = Pipeline(
        [
            ("reduce_dim", "passthrough"),
            ("classifier", clf),
        ]
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="weighted"),
        "recall": make_scorer(recall_score, average="weighted"),
        "f1": make_scorer(f1_score, average="weighted"),
    }

    cv_results = cross_validate(
        pipeline_clf,
        train_features,
        train_labels,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
        verbose=1,
    )

    pipeline = pipeline_clf.set_params(**params).fit(train_features, train_labels)
    test_pred = pipeline.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_pred)
    test_precision = precision_score(test_labels, test_pred, average="weighted")
    test_recall = recall_score(test_labels, test_pred, average="weighted")
    test_f1 = f1_score(test_labels, test_pred, average="weighted")

    results = pd.Series(
        {
            "Model": clf.__class__.__qualname__,
            "Validation Accuracy": (
                np.mean(cv_results["test_accuracy"]),
                np.std(cv_results["test_accuracy"]),
            ),
            "Test Accuracy": test_accuracy,
            "Validation Precision": (
                np.mean(cv_results["test_precision"]),
                np.std(cv_results["test_precision"]),
            ),
            "Test Precision": test_precision,
            "Validation Recall": (
                np.mean(cv_results["test_recall"]),
                np.std(cv_results["test_recall"]),
            ),
            "Test Recall": test_recall,
            "Train F1": (np.mean(cv_results["test_f1"]), np.std(cv_results["test_f1"])),
            "Test F1": test_f1,
            "Train Fit time": (
                np.mean(cv_results["fit_time"]),
                np.std(cv_results["fit_time"]),
            ),
        }
    )
    results.name = experiment
    return results


if __name__ == "__main__":
    import argparse

    np.random.seed(0)

    parser = argparse.ArgumentParser(
        prog="Compute metrics",
        description="This program computes cross validation with the best model for each task",
    )
    parser.add_argument(
        "train_features",
        type=Path,
        help="the path to the radiomics features CSV training set",
    )
    parser.add_argument(
        "train_labels",
        type=Path,
        help="the path to the labels CSV training set",
    )
    parser.add_argument(
        "test_features",
        type=Path,
        help="the path to the radiomics features CSV test set",
    )
    parser.add_argument(
        "test_labels",
        type=Path,
        help="the path to the labels CSV test set",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="the path to the output CSV",
    )
    args = parser.parse_args()

    train_features_path = args.train_features
    train_labels_path = args.train_labels
    test_features_path = args.test_features
    test_labels_path = args.test_labels
    output = args.output

    experiments = {
        "5 levels/L5-S": {
            "classifier": LogisticRegression(),
            "params": {
                "reduce_dim": PCA(n_components=0.95, random_state=0),
                "classifier__solver": "sag",
                "classifier__penalty": "l2",
                "classifier__max_iter": 200,
                "classifier__C": 1.0,
            },
            "disc": 1,
        },
        "5 levels/L4-L5": {
            "classifier": GradientBoostingClassifier(),
            "params": {
                "classifier__subsample": 1.0,
                "classifier__n_estimators": 200,
                "classifier__max_features": "sqrt",
                "classifier__max_depth": 4,
                "classifier__learning_rate": 0.1,
            },
            "disc": 2,
        },
        "5 levels/L3-L4": {
            "classifier": MLPClassifier(),
            "params": {
                "reduce_dim": mRMRFeatureReduction(K=20),
                "classifier__solver": "adam",
                "classifier__learning_rate": "adaptive",
                "classifier__hidden_layer_sizes": (50, 50),
                "classifier__alpha": 0.01,
                "classifier__activation": "relu",
            },
            "disc": 3,
        },
        "5 levels/L2-L3": {
            "classifier": GradientBoostingClassifier(),
            "params": {
                "reduce_dim": mRMRFeatureReduction(K=20),
                "classifier__subsample": 0.7,
                "classifier__n_estimators": 100,
                "classifier__max_features": "log2",
                "classifier__max_depth": 2,
                "classifier__learning_rate": 0.1,
            },
            "disc": 4,
        },
        "5 levels/L1-L2": {
            "classifier": MLPClassifier(),
            "params": {
                "reduce_dim": mRMRFeatureReduction(K=20),
                "classifier__solver": "adam",
                "classifier__learning_rate": "constant",
                "classifier__hidden_layer_sizes": (50, 50),
                "classifier__alpha": 0.01,
                "classifier__activation": "tanh",
            },
            "disc": 5,
        },
        "5 levels/ALL": {
            "classifier": LogisticRegression(),
            "params": {
                "classifier__solver": "liblinear",
                "classifier__penalty": "l2",
                "classifier__max_iter": 300,
                "classifier__C": 10.0,
            },
        },
        "4 levels/L5-S": {
            "classifier": SVC(kernel="linear"),
            "params": {
                "reduce_dim": PCA(n_components=0.95, random_state=0),
                "classifier__gamma": "scale",
                "classifier__C": 1,
            },
            "disc": 1,
        },
        "4 levels/L4-L5": {
            "classifier": GradientBoostingClassifier(),
            "params": {
                "classifier__subsample": 1.0,
                "classifier__n_estimators": 500,
                "classifier__max_features": "sqrt",
                "classifier__max_depth": 2,
                "classifier__learning_rate": 0.2,
            },
            "disc": 2,
        },
        "4 levels/L3-L4": {
            "classifier": MLPClassifier(),
            "params": {
                "reduce_dim": mRMRFeatureReduction(K=20),
                "classifier__solver": "adam",
                "classifier__learning_rate": "constant",
                "classifier__hidden_layer_sizes": (50, 50),
                "classifier__alpha": 0.0001,
                "classifier__activation": "relu",
            },
            "disc": 3,
        },
        "4 levels/L2-L3": {
            "classifier": GradientBoostingClassifier(),
            "params": {
                "reduce_dim": mRMRFeatureReduction(K=10),
                "classifier__subsample": 0.9,
                "classifier__n_estimators": 500,
                "classifier__max_features": "sqrt",
                "classifier__max_depth": 4,
                "classifier__learning_rate": 0.05,
            },
            "disc": 4,
        },
        "4 levels/L1-L2": {
            "classifier": MLPClassifier(),
            "params": {
                "reduce_dim": PCA(n_components=0.99, random_state=0),
                "classifier__solver": "adam",
                "classifier__learning_rate": "invscaling",
                "classifier__hidden_layer_sizes": (50,),
                "classifier__alpha": 0.0001,
                "classifier__activation": "tanh",
            },
            "disc": 5,
        },
        "4 levels/ALL": {
            "classifier": SVC(kernel="linear"),
            "params": {"classifier__gamma": "scale", "classifier__C": 1},
        },
    }

    results = []
    for key, config in experiments.items():
        train_labels = pd.read_csv(train_labels_path, index_col="ID")
        train_features = pd.read_csv(train_features_path, index_col="ID")
        test_labels = pd.read_csv(test_labels_path, index_col="ID")
        test_features = pd.read_csv(test_features_path, index_col="ID")
        if "ALL" not in key:
            train_labels = train_labels.loc[
                train_labels.index.str.endswith(str(config["disc"]))
            ]
            train_features = train_features.loc[
                train_features.index.str.endswith(str(config["disc"]))
            ]
            test_labels = test_labels.loc[
                test_labels.index.str.endswith(str(config["disc"]))
            ]
            test_features = test_features.loc[
                test_features.index.str.endswith(str(config["disc"]))
            ]
        if "4" in key:
            train_labels[train_labels == 1] = 2
            test_labels[test_labels == 1] = 2
        classifier = config["classifier"]
        params = config["params"]
        result = compute_metrics(
            key,
            classifier,
            params,
            train_features,
            train_labels,
            test_features,
            test_labels,
        )
        print(result)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output)
