from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transforms import CorrelationFeatureReduction, VarianceFeatureReduction
from utils import get_labels_and_features, get_labels_and_features_all_discs


def random_search_cv(experiment, clf, search_grid, features, labels):
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

    rs_clf = RandomizedSearchCV(
        pipeline_clf,
        search_grid,
        cv=skf,
        scoring="f1_weighted",
        n_iter=100,
        n_jobs=-1,
        random_state=0,
        verbose=1,
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

    random_forest_distribution = {
        "classifier__n_estimators": [10, 50, 100, 200, 500],
        "classifier__max_features": ["auto", "sqrt", "log2"],
        "classifier__max_depth": [4, 5, 6, 7, 8],
        "classifier__criterion": ["gini", "entropy"],
    }

    gradient_boosting_distribution = {
        "classifier__n_estimators": [50, 100, 200, 500],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__max_features": ["auto", "sqrt", "log2"],
    }

    extra_trees_distribution = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt", "log2"],
    }

    mlp_distribution = {
        "classifier__hidden_layer_sizes": [(50,), (100,), (200,)],
        "classifier__activation": ["relu", "tanh"],
        "classifier__solver": ["lbfgs", "adam"],
        "classifier__alpha": [0.0001, 0.001],
        "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
    }

    experiments = {
        "5 levels/L5-S": {
            "classifier": GradientBoostingClassifier,
            "distribution": gradient_boosting_distribution,
            "disc": 1,
        },
        "5 levels/L4-L5": {
            "classifier": RandomForestClassifier,
            "distribution": random_forest_distribution,
            "disc": 2,
        },
        "5 levels/L3-L4": {
            "classifier": ExtraTreesClassifier,
            "distribution": extra_trees_distribution,
            "disc": 3,
        },
        "5 levels/L2-L3": {
            "classifier": RandomForestClassifier,
            "distribution": random_forest_distribution,
            "disc": 4,
        },
        "5 levels/L1-L2": {
            "classifier": RandomForestClassifier,
            "distribution": random_forest_distribution,
            "disc": 5,
        },
        "5 levels/ALL": {
            "classifier": ExtraTreesClassifier,
            "distribution": extra_trees_distribution,
        },
        "4 levels/L5-S": {
            "classifier": GradientBoostingClassifier,
            "distribution": gradient_boosting_distribution,
            "disc": 1,
        },
        "4 levels/L4-L5": {
            "classifier": RandomForestClassifier,
            "distribution": random_forest_distribution,
            "disc": 2,
        },
        "4 levels/L3-L4": {
            "classifier": ExtraTreesClassifier,
            "distribution": extra_trees_distribution,
            "disc": 3,
        },
        "4 levels/L2-L3": {
            "classifier": MLPClassifier,
            "distribution": mlp_distribution,
            "disc": 4,
        },
        "4 levels/L1-L2": {
            "classifier": RandomForestClassifier,
            "distribution": random_forest_distribution,
            "disc": 5,
        },
        "4 levels/ALL": {
            "classifier": RandomForestClassifier,
            "distribution": random_forest_distribution,
        },
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
        classifier = config["classifier"]()
        result = random_search_cv(
            key, classifier, config["distribution"], features, labels
        )
        print(result)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output)
