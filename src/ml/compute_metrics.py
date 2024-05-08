from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sklearn.preprocessing import StandardScaler
from transforms import CorrelationFeatureReduction, VarianceFeatureReduction
from utils import get_labels_and_features, get_labels_and_features_all_discs


def cv(experiment, clf, features, labels):
    # Create a stratified 5-fold cross-validation object
    skf = StratifiedKFold(n_splits=5)

    # Perform cross-validation
    pipeline_clf = Pipeline(
        [
            ("variancethreshold", VarianceFeatureReduction(threshold=0.05)),
            ("correlationreduction", CorrelationFeatureReduction()),
            ("scaler", StandardScaler()),
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
        features,
        labels,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
        verbose=1,
    )

    results = pd.Series(
        {
            "Model": clf.__class__.__qualname__,
            "Accuracy": (
                np.mean(cv_results["test_accuracy"]),
                np.std(cv_results["test_accuracy"]),
            ),
            "Precision": (
                np.mean(cv_results["test_precision"]),
                np.std(cv_results["test_precision"]),
            ),
            "Recall": (
                np.mean(cv_results["test_recall"]),
                np.std(cv_results["test_recall"]),
            ),
            "F1": (np.mean(cv_results["test_f1"]), np.std(cv_results["test_f1"])),
            "Fit time": (
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
        "img_relation",
        type=Path,
        help="the path to the image relation CSV",
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

    img_relation_path = args.img_relation
    features_path = args.features
    labels_path = args.labels
    output = args.output

    experiments = {
        "5 levels/L5-S": {
            "classifier": GradientBoostingClassifier(
                n_estimators=200, max_features="log2", max_depth=2, learning_rate=0.1
            ),
            "disc": 1,
        },
        "5 levels/L4-L5": {
            "classifier": RandomForestClassifier(
                n_estimators=50, max_features="sqrt", max_depth=6, criterion="entropy"
            ),
            "disc": 2,
        },
        "5 levels/L3-L4": {
            "classifier": ExtraTreesClassifier(),
            "disc": 3,
        },
        "5 levels/L2-L3": {
            "classifier": RandomForestClassifier(
                max_features="log2", max_depth=7, criterion="entropy"
            ),
            "disc": 4,
        },
        "5 levels/L1-L2": {
            "classifier": RandomForestClassifier(
                max_features="sqrt", max_depth=7, criterion="gini"
            ),
            "disc": 5,
        },
        "5 levels/ALL": {"classifier": ExtraTreesClassifier()},
        "4 levels/L5-S": {
            "classifier": GradientBoostingClassifier(
                subsample=0.8,
                n_estimators=200,
                max_features="sqrt",
                max_depth=4,
                learning_rate=0.1,
            ),
            "disc": 1,
        },
        "4 levels/L4-L5": {
            "classifier": RandomForestClassifier(max_depth=7),
            "disc": 2,
        },
        "4 levels/L3-L4": {"classifier": ExtraTreesClassifier(), "disc": 3},
        "4 levels/L2-L3": {"classifier": MLPClassifier(), "disc": 4},
        "4 levels/L1-L2": {"classifier": RandomForestClassifier(), "disc": 5},
        "4 levels/ALL": {"classifier": RandomForestClassifier()},
    }

    results = []
    for key, config in experiments.items():
        if "ALL" not in key:
            labels, features = get_labels_and_features(
                img_relation_path, labels_path, features_path, label=config["disc"]
            )
        else:
            labels, features = get_labels_and_features_all_discs(
                img_relation_path, labels_path, features_path
            )
        if "4" in key:
            labels.loc[labels == 1] = 2
        classifier = config["classifier"]
        result = cv(key, classifier, features, labels)
        print(result)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output)
