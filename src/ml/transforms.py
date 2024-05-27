import numpy as np
import pandas as pd
import pingouin as pg
from mrmr import mrmr_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


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
        X_ = X_.loc[:, self.to_keep]
        return X_


class ICCFeatureReduction(BaseEstimator, TransformerMixin):
    """
    A transformer class for reducing features based on Intraclass Correlation (ICC).

    Parameters:
    -----------
    threshold : float, optional (default=0.8)
        The threshold above which features will be removed.
    """

    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.to_keep = None

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.

        Parameters:
        -----------
        X : tuple of pandas DataFrame
            The different `variations` of input data (e.g. radiomic features from perturbed masks).

        Returns:
        --------
        self : ICCFeatureReduction
            The fitted transformer object.

        """

        # Initialize an empty flag vector
        to_keep = []

        for feature in X[0].columns:
            # Concatenate feature values vertically
            feature_data = pd.concat(
                [dataframe[[feature]] for dataframe in X],
                axis=0,
                ignore_index=False,
            )

            # Append patient/repetition information
            # Create a repetition/patients column
            result_array = np.repeat(
                [1, 2, 3, 4],
                [len(dataframe) for dataframe in X],
            )
            feature_data["Repetition"] = result_array
            feature_data["Patients"] = pd.factorize(feature_data.index)[0]
            feature_data = feature_data.rename(columns={feature: "FeatureValue"})

            # Compute ICC
            icc_result = pg.intraclass_corr(
                data=feature_data,
                targets="Patients",
                raters="Repetition",
                ratings="FeatureValue",
            )
            # Extract ICC value
            icc_value = icc_result["ICC"].iloc[
                1
            ]  # ICC2: A random sample of raters rate each target. Measure of absolute agreement.

            # Check if ICC is greater than the threshold
            if icc_value > self.threshold:
                to_keep.append(True)
            else:
                to_keep.append(False)
        self.to_keep = to_keep
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by keeping 'invariant' features.

        Parameters:
        -----------
        X : pandas DataFrame
            The input data.

        Returns:
        --------
        X_ : pandas DataFrame
            The transformed data with 'invariant' features.

        """
        X_ = X.copy()
        X_ = X_.loc[:, self.to_keep]
        return X_


class mRMRFeatureReduction(BaseEstimator, TransformerMixin):
    """
    A transformer class for reducing features based on minimum Redundancy - Maximum Relevance.

    Parameters:
    -----------
    K : int, optional (default=10)
        The maximum number of features to keep.
    """

    def __init__(self, K=10):
        self.K = K
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
        self : mRMRFeatureReduction
            The fitted transformer object.

        """

        self.to_keep = mrmr_classif(X=X, y=y, K=self.K, show_progress=False)
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by removing selected features.

        Parameters:
        -----------
        X : pandas DataFrame
            The input data.

        Returns:
        --------
        X_ : pandas DataFrame
            The transformed data.

        """
        X_ = X.copy()
        X_ = X_.loc[:, self.to_keep]
        return X_
