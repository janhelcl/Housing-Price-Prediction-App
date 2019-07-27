"""
Collection of custom scikit-learn compatible transformers.
"""
from typing import List, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from processing.exceptions import InvalidInputError


class TemporalDifferenceTransformer(BaseEstimator, TransformerMixin):
    """Temporal variable calculator.
    
    Calculates difference in time from a single reference variable.
    
    :param temp_vars: List of temporal variables
    :param reference_var: Reference variable
    """

    def __init__(self,
                 temp_vars: List[str],
                 reference_var: str
                 ):

        self.temp_vars = temp_vars
        self.reference_var = reference_var

    def fit(self, X, y=None):
        "For compatibility only"
        return self

    def transform(self,
                  X: pd.DataFrame
                  ) -> pd.DataFrame:
        """
        Calculates difference in time from a single reference variable.
        
        :param X: pd.DataFrame of model predictors
        
        :returns: Transformed data
        """
        X = X.copy()
        for feature in self.temp_vars:
            X[feature] = X[self.reference_var] - X[feature]

        return X
    
    
class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drops selected columns.
    
    Drops selected columns inside of a scikit-learn pipline. Useful for example
    for features used for feature engineering in previous steps that are
    no longer needed.
    
    :param vars_to_drop: List of features to drop
    """ 

    def __init__(self,
                 vars_to_drop: List[str]
                 ):

        self.vars_to_drop = vars_to_drop

    def fit(self, X, y=None):
        "For compatibility only"
        return self

    def transform(self, X):
        """Drops selected columns
        
        :param X: pd.DataFrame of model predictors
        
        :returns: Data without unwanted features
        """
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder
    
    Merge all rare labels (proportin below selected tolerance) into one
    category 'rare'. Previously unseen labels during transform are also
    encoded as 'rare'.

    :param variables: List of variables to be encoded
    :param tol: Tolerance level, lables below this proportion will be merged
    """

    def __init__(self, 
                 variables: List[str],
                 tol:float = 0.0
                 ):
        
        self.variables = variables
        self.tol = tol

    def fit(self,
            X: pd.DataFrame,
            y=None
            ) -> 'RareLabelEncoder':
        """Finds frequent categories
        
        :param X: pd.DataFrame of model predictors
        :param y: For compatibility only
        
        :returns: self
        """
        self.frequent_labels_ = {}
        for var in self.variables:
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            self.frequent_labels_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self,
                  X: pd.DataFrame
                  ) -> pd.DataFrame:
        """Encodes the infrequent labels to 'rare'
        
        :param X: pd.DataFrame of model predictors
        
        :returns: Encoded data
        """
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.frequent_labels_[feature]), X[feature], 'rare')

        return X


class UnivariateTransformer(BaseEstimator, TransformerMixin):
    """Applies provided function to the selected columns
    
    :param variables: List of variables to be encoded
    :param func: function to be applied (must support pd.Series as input)
    """

    def __init__(self,
                 variables: List[str],
                 func: Callable
                 ):
    
        self.variables = variables
        self.func = func
        
    def fit(self, X, y=None):
        "For compatibility only"
        return self

    def transform(self,
                  X: pd.DataFrame
                  ) -> pd.DataFrame:
        """Applies provided function to the selected columns
        
        :param X: pd.DataFrame of model predictors
        
        :returns: Transformed data
        """
        X = X.copy()
        for feature in self.variables:
            try:
                X[feature] = self.func(X[feature])
            except Exception as error:
                raise InvalidInputError(
                        ("Provided function failed to transform"
                         f" column {feature}.")
                                        ) from error
        return X