from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from transforms import (
    CorrelationFeatureReduction,
    mRMRFeatureReduction,
    VarianceFeatureReduction,
)

MRMR_FEATURES_OPTIONS = [5, 10, 20]
PCA_VARIANCE_OPTIONS = [0.95, 0.99]
BASE_PARAM_GRID = [
    {},
    {
        "reduce_dim": [CorrelationFeatureReduction()],
    },
    {
        "reduce_dim": [mRMRFeatureReduction()],
        "reduce_dim__K": MRMR_FEATURES_OPTIONS,
    },
    {
        "reduce_dim": [PCA(random_state=0)],
        "reduce_dim__n_components": PCA_VARIANCE_OPTIONS,
    },
    {
        "reduce_dim": [VarianceFeatureReduction()],
    },
    {
        "reduce_dim": [
            make_pipeline(VarianceFeatureReduction(), CorrelationFeatureReduction())
        ],
    },
    {
        "reduce_dim": [
            make_pipeline(VarianceFeatureReduction(), mRMRFeatureReduction())
        ],
        "reduce_dim__mrmrfeaturereduction__K": MRMR_FEATURES_OPTIONS,
    },
    {
        "reduce_dim": [make_pipeline(VarianceFeatureReduction(), PCA(random_state=0))],
        "reduce_dim__pca__n_components": PCA_VARIANCE_OPTIONS,
    },
]


def test_multiple_models(experiment, features, labels):
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "LinearSVM": SVC(kernel="linear"),
        "RadialSVM": SVC(kernel="rbf"),
        "Logistic Regression": LogisticRegression(),
        "Stochastic Gradient Descent": SGDClassifier(),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Multilayer Perceptron": MLPClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
    }
    results = []
    for name, clf in classifiers.items():
        pipeline = Pipeline(
            [
                ("reduce_dim", "passthrough"),
                ("classifier", clf),
            ]
        )
        grid_search = GridSearchCV(
            pipeline,
            BASE_PARAM_GRID,
            cv=StratifiedKFold(n_splits=10),
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=2,
        )
        grid_results = grid_search.fit(features, labels)
        clf_results = pd.DataFrame(data=grid_results.cv_results_)
        clf_results["Image Type"] = "all"

        for image_type in ("log", "wavelet", "original"):
            X_train = features.loc[
                :,
                features.columns.str.contains(image_type)
                & ~features.columns.str.contains("diagnostics"),
            ].copy()
            grid_results = grid_search.fit(X_train, labels)

            results_aux = pd.DataFrame(data=grid_results.cv_results_)
            results_aux["Image Type"] = image_type

            clf_results = pd.concat([clf_results, results_aux])

        clf_results["Model"] = name

        results.append(clf_results)

    results = pd.concat(results)

    results["Experiment"] = experiment

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
        labels = pd.read_csv(labels_path, index_col="ID")
        features = pd.read_csv(features_path, index_col="ID")
        if "ALL" not in key:
            labels = labels.loc[labels.index.str.endswith(str(config["disc"]))]
            features = features.loc[features.index.str.endswith(str(config["disc"]))]
        if "4" in key:
            labels[labels == 1] = 2
        result = test_multiple_models(key, features, labels)
        results.append(result)

    results_df = pd.concat(results)
    results_df.to_csv(output, index=False)
