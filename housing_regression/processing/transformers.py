"""
Collection of custom scikit-learn compatible transformers.
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from exceptions import InvalidInputError


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
                 vars_to_drop: List[str] = None
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
    
    :param tol: Tolerance level, lables below this proportion will be merged
    :param variables: List of variables to be encoded
    """

    def __init__(self, 
                 tol:float = 0.0,
                 variables: List[str] = None
                 ):
        
        self.tol = tol
        self.variables = variables

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