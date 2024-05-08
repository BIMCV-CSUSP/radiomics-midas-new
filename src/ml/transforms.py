import numpy as np
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
        X_ = X_.iloc[:, self.to_keep]
        return X_
