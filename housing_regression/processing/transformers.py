"""
Collection of custom scikit-learn compatible transformers.

For convenience with pipeline configuration made to work with pd.DataFrames.
Can be turned into separate package and specified as a dependency once stable.
"""
from typing import List, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from processing.exceptions import InvalidInputError


class ColumnTransformerDF(ColumnTransformer):
    """Extention of sklearn.compose.ColumnTransformer to return pd.DataFrame
    
    Applies transformers to columns of pandas DataFrame. This estimator allows 
    different columns or column subsets of the input to be transformed 
    separately and the features generated by each transformer will be 
    concatenated to form a single feature space. This is useful for 
    heterogeneous or columnar data, to combine several feature extraction 
    mechanisms or transformations into a single transformer.
    
    Unlike the original sklearn implementation, there is one more assumption:
    there must be 1:1 correspondence between original and transformed columns.
    Will not work for example with sklearn.preprocessing.OneHotEncoder.
    
    :param transformers: List of (name, transformer, column(s)) tuples 
        specifying the transformer objects to be applied to subsets of the data
    :param remainder: what to do with remaining columns - 'drop', 'passthrough'
        or an estimator
    """
    
    def fit(self,
            X: pd.DataFrame,
            y=None
            ) -> 'ColumnTransformerDF':
        """Fit all transformers using X
        
        :param X: pd.DataFrame of model predictors
        :param y: For compatibility only
        
        :returns: self
        """
        self.columns_ = X.columns
        return super().fit(X, y=y)
    
    def transform(self,
                  X: pd.DataFrame
                  ) -> pd.DataFrame:
        """Transform X separately by each transformer, concatenate results
        
        :param X: pd.DataFrame of model predictors
        
        :returns: Transformed data
        """
        X = X.copy()
        return pd.DataFrame(data=super().transform(X),
                            columns=self.columns_)
        
    def fit_transform(self,
                      X:pd.DataFrame,
                      y=None) -> pd.DataFrame:
        """Fit and then transform
        
        :param X: pd.DataFrame of model predictors
        
        :returns: Transformed data
        """
        X = X.copy()
        return pd.DataFrame(data=super().fit_transform(X, y),
                            columns=X.columns)
  
 
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


# TODO: add option to add new column insted of transforming the old in place
class BivariateTransformer(BaseEstimator, TransformerMixin):
    """Transforms all selected features using another feature
    
    Apllies provided function with the signature: f(varible, reference_var).
    For example: if provided 'lambda a, b: a / b' then all provided variables
    will be transformed into ratios with reference_var as denominator
    
    :param variables: List of temporal variables
    :param reference_var: Reference variable
    :param func: Callable combining two pd.Series: f(varible, reference_var)
    """

    def __init__(self,
                 variables: List[str],
                 reference_var: str,
                 func: str = 'ratio'
                 ):

        self.variables = variables
        self.reference_var = reference_var
        self.func = func

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
        for feature in self.variables:
            X[feature] = self.func(X[feature].copy(), X[self.reference_var].copy())

        return X
    
    
class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drops selected columns.
    
    Drops selected columns inside of a scikit-learn pipline. Useful for example
    for features used for feature engineering in previous steps that are
    no longer needed. Typically used after BivariateTransformer.
    
    :param vars_to_drop: List of features to drop
    """ 

    def __init__(self,
                 vars_to_drop: List[str]
                 ):

        self.vars_to_drop = vars_to_drop

    def fit(self, X, y=None):
        "For compatibility only"
        return self

    def transform(self,
                  X: pd.DataFrame
                  ) -> pd.DataFrame:
        """Drops selected columns
        
        :param X: pd.DataFrame of model predictors
        
        :returns: Data without unwanted features
        """
        X = X.copy()
        return X.drop(self.variables, axis=1)


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
                 tol:float = 0.05
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
    