from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transforms import CorrelationFeatureReduction, VarianceFeatureReduction
from utils import get_labels_and_features, get_labels_and_features_all_discs


def test_multiple_models(experiment, features, labels):
    # Create a stratified 5-fold cross-validation object
    skf = StratifiedKFold(n_splits=5)

    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(),
        "Stochastic Gradient Descent": SGDClassifier(),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Multilayer Perceptron": MLPClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
    }

    f1_scores = {}
    for name, clf in classifiers.items():
        pipeline = Pipeline(
            [
                ("variancethreshold", VarianceFeatureReduction(threshold=0.05)),
                ("correlationreduction", CorrelationFeatureReduction()),
                ("scaler", StandardScaler()),
                ("classifier", clf),
            ]
        )
        scores = cross_val_score(
            pipeline,
            features,
            labels,
            cv=skf,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )
        f1_scores[name] = scores.mean()

    # Select the classifier with the highest F1 score
    best_classifier = max(f1_scores, key=f1_scores.get)  # type: ignore

    results = pd.Series(
        {
            "Best classifier": best_classifier,
            "F1": f1_scores[best_classifier],
        }
    )
    results.name = experiment
    return results


if __name__ == "__main__":
    import argparse

    np.random.seed(0)

    parser = argparse.ArgumentParser(
        prog="Test models from scikit-learn",
        description="This program trains and evaluates several models from scikit-learn"
        "(with default parameters) on each task, and returns which performs best.",
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
        "5 levels/L5-S": {"disc": 1},
        "5 levels/L4-L5": {"disc": 2},
        "5 levels/L3-L4": {"disc": 3},
        "5 levels/L2-L3": {"disc": 4},
        "5 levels/L1-L2": {"disc": 5},
        "5 levels/ALL": {},
        "4 levels/L5-S": {"disc": 1},
        "4 levels/L4-L5": {"disc": 2},
        "4 levels/L3-L4": {"disc": 3},
        "4 levels/L2-L3": {"disc": 4},
        "4 levels/L1-L2": {"disc": 5},
        "4 levels/ALL": {},
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
        result = test_multiple_models(key, features, labels)
        print(result)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output)
