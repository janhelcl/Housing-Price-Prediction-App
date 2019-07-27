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

    def fit(self,
            X: pd.DataFrame,
            y=None
            ) -> 'TemporalDifferenceTransformer':
        "For compatibility only"
        return self

    def transform(self,
                  X: pd.DataFrame
                  ) -> pd.DataFrame:
        """
        Calculates difference in time from a single reference variable.
        
        :param X: pd.DataFrame of model predictors
        """
        X = X.copy()
        for feature in self.temp_vars:
            X[feature] = X[self.reference_var] - X[feature]

        return X
    
    
class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drops selected columns.
    
    Drops selected columns inside of a scikit-learn pipline. Useful for example
    for features used fo feature engineering in previous steps that are
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
        """
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X
