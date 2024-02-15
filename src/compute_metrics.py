from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class VarianceFeatureReduction(BaseEstimator, TransformerMixin):
    """
    VarianceFeatureReduction is a transformer that reduces the feature space by removing features with low variance.

    Parameters:
    -----------
    threshold : float, optional (default=0.05)
        The threshold below which features will be removed.
    """

    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.selector = None

    def fit(self, X, y=None):
        """
        Fit the VarianceFeatureReduction transformer to the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(X)
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by removing features with low variance.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns:
        --------
        X_ : array-like, shape (n_samples, n_selected_features)
            The transformed data with low variance features removed.
        """
        X_ = X.copy()
        X_ = X_.loc[:, self.selector.get_support()]
        return X_


class CorrelationFeatureReduction(BaseEstimator, TransformerMixin):
    """
    A transformer class for reducing features based on correlation.

    Parameters:
    -----------
    threshold : float, optional (default=0.8)
        The threshold above which features will be removed.
    """

    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.corr_matrix_var = None
        self.to_keep = None

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.

        Parameters:
        -----------
        X : pandas DataFrame
            The input data.

        Returns:
        --------
        self : CorrelationFeatureReduction
            The fitted transformer object.

        """
        self.corr_matrix_var = X.corr(
            method="spearman"
        ).abs()  # absolute correlation matrix

        # Initialize the flag vector with True values
        self.to_keep = np.full((self.corr_matrix_var.shape[1]), True, dtype=bool)

        for i in range(self.corr_matrix_var.shape[1]):
            for j in range(i + 1, self.corr_matrix_var.shape[1]):
                if (
                    self.to_keep[i]
                    and self.corr_matrix_var.iloc[i, j] >= self.threshold
                ):
                    if self.to_keep[j]:
                        self.to_keep[j] = False
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by removing highly correlated features.

        Parameters:
        -----------
        X : pandas DataFrame
            The input data.

        Returns:
        --------
        X_ : pandas DataFrame
            The transformed data with highly correlated features removed.

        """
        X_ = X.copy()
        X_ = X_.iloc[:, self.to_keep]
        return X_


def build_dataframe_from_csv(
    rater: str = "900", from_image: str = "t2w"
) -> pd.DataFrame:
    labels_df = pd.read_csv(
        root_dir.joinpath("data", f"midasdisclabels{rater}.csv"), sep=","
    )
    labels_df.dropna(inplace=True)
    labels_df.rename(
        columns={"subject_ID": "Subject_XNAT", "ID": "Session_XNAT"}, inplace=True
    )

    midas_img_relation = pd.read_csv(
        root_dir.joinpath("data", "filtered_midas900_t2w.csv"), sep=","
    )
    midas_img_relation["Subject_MIDS"] = midas_img_relation["Image"].map(
        lambda x: x.split("/")[8]
    )
    midas_img_relation["Session_MIDS"] = midas_img_relation["Image"].map(
        lambda x: x.split("/")[9]
    )
    midas_img_relation["Subject_XNAT"] = midas_img_relation["Subject_MIDS"].map(
        lambda x: f"ceibcs_S{int(x.split('sub-S')[1])}"
    )
    midas_img_relation["Session_XNAT"] = midas_img_relation["Session_MIDS"].map(
        lambda x: f"ceibcs_E{int(x.split('ses-E')[1])}"
    )

    id_labels = labels_df.merge(midas_img_relation, on=["Subject_XNAT", "Session_XNAT"])
    id_labels.rename(
        columns={
            "L5-S": "1",
            "L4-L5": "2",
            "L3-L4": "3",
            "L2-L3": "4",
            "L1-L2": "5",
        },
        inplace=True,
    )

    radiomic_features = pd.read_csv(
        root_dir.joinpath("data", f"filtered_midas900_{from_image}_radiomics.csv"),
        sep=",",
    )
    radiomic_features.rename(columns={"Unnamed: 0": "ID"}, inplace=True)

    return id_labels.merge(radiomic_features, on="ID")


def get_labels_and_features(
    rater: str = "900", label: int = 1, from_image: str = "t2w"
) -> tuple:
    """
    Reads a CSV file from the given label and returns the labels and features as separate dataframes.

    :param rater: The rater identifier. Default is "900".
    :type rater: str
    :param label: A number from 1 to 5 indicating the disc of interest. Default is 1.
    :type label: bool
    :return: A tuple containing the labels and features.
    :rtype: tuple
    """

    data = build_dataframe_from_csv(rater=rater, from_image=from_image)

    data = data.rename(columns={str(label): f"label{label}", "ID": f"label{label}ID"})
    columns_mask = data.columns.str.contains(
        f"label{label}"
    ) & ~data.columns.str.contains("Configuration")
    data = data.loc[:, columns_mask]
    data = data.rename(columns={f"label{label}": "label", f"label{label}ID": "ID"})

    label_data = data.dropna(axis=0, how="any")
    label_data = label_data.loc[label_data["label"] != 0]
    label_data["ID"] = label_data["ID"].map(lambda x: x + str(label))
    label_data = label_data.set_index("ID")
    labels = label_data["label"]
    features = label_data[
        label_data.select_dtypes(include="number").columns.tolist()
    ].drop(columns="label")
    return labels, features


def get_labels_and_features_all_discs(
    rater: str = "900", verbose: bool = False, from_image: str = "t1w_t2w"
) -> tuple:
    """
    Get labels and features for all discs.

    :param rater: The rater identifier. Default is "900".
    :type rater: str
    :param verbose: Whether to print additional information and plot the label distribution. Default is False.
    :type verbose: bool
    :return: A tuple containing the labels and features.
    :rtype: tuple
    """
    features = []
    labels = []
    for label in range(1, 6):
        labels_i, features_i = get_labels_and_features(
            rater=rater, label=label, from_image=from_image
        )
        labels.append(labels_i)
        features_i = features_i.rename(
            columns={
                name: name.replace(f"label{label}_", "")
                for name in features_i.columns.to_list()
            }
        )
        features.append(features_i)
    features = pd.concat(features, axis=0)
    labels = pd.concat(labels, axis=0)
    if verbose:
        print(f"Labels shape: {labels.shape}, Features shape: {features.shape}")
    return labels, features


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
    np.random.seed(0)

    root_dir = Path(__file__).resolve().parents[1]

    experiments = {
        "5 levels/L5-S": {"classifier": GradientBoostingClassifier, "disc": 1},
        "5 levels/L4-L5": {"classifier": RandomForestClassifier, "disc": 2},
        "5 levels/L3-L4": {"classifier": GradientBoostingClassifier, "disc": 3},
        "5 levels/L2-L3": {"classifier": RandomForestClassifier, "disc": 4},
        "5 levels/L1-L2": {"classifier": RandomForestClassifier, "disc": 5},
        "5 levels/ALL": {"classifier": GradientBoostingClassifier},
        "4 levels/L5-S": {"classifier": GradientBoostingClassifier, "disc": 1},
        "4 levels/L4-L5": {"classifier": RandomForestClassifier, "disc": 2},
        "4 levels/L3-L4": {"classifier": RandomForestClassifier, "disc": 3},
        "4 levels/L2-L3": {"classifier": RandomForestClassifier, "disc": 4},
        "4 levels/L1-L2": {"classifier": RandomForestClassifier, "disc": 5},
        "4 levels/ALL": {"classifier": GradientBoostingClassifier},
    }

    results = []

    for key, config in experiments.items():
        if "ALL" not in key:
            labels, features = get_labels_and_features(
                rater="JDCarlos", label=config["disc"], from_image="t1w_t2w"
            )
        else:
            labels, features = get_labels_and_features_all_discs(
                rater="JDCarlos", from_image="t1w_t2w"
            )
        classifier = config["classifier"]()
        result = cv(key, classifier, features, labels)
        print(result)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(root_dir.joinpath("data", "results", "cv_results.csv"))
