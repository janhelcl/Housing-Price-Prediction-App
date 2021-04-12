"""
Test the transformers
"""
import pickle
import tempfile
import functools
import math
import sys
sys.path.append('..')

import pytest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import housing_regression.processing.transformers as tran
from housing_regression.processing.exceptions import InvalidInputError


TEST_DATA_SIZE = 1000

@pytest.fixture(scope='module')
def data():
    data = {
    'num1': np.random.uniform(size=TEST_DATA_SIZE),
    'num2': np.random.uniform(size=TEST_DATA_SIZE),
    'cat': np.random.choice(['a', 'b', 'c'], size=TEST_DATA_SIZE)
    }
    data['cat'][0] = 'd' # add a rare category for tran.RareLabelEncoder
    return pd.DataFrame.from_dict(data)


# dummy functions to test tran.BivariateTransformer
def diff(a, b):
    return a - b

def add(a, b):
    return a + b

def product(a, b):
    return a * b

def leave_first(a, b):
    return a


class TestGeneralProperties():
    """Test common properties of all transformers
    """

    TRANSFORMERS = [
            tran.ColumnTransformerDF([('scaler', StandardScaler(), ['num1'])],
                                      remainder='passthrough'),
            tran.UnivariateTransformer(variables=['num2'], func=np.log),
            tran.BivariateTransformer(variables=['num1'],
                                      reference_var='num2',
                                      func=diff
                                      ),
            tran.FeatureDropper(vars_to_drop=['num1', 'num2']),
            tran.RareLabelEncoder(variables=['cat'])
            ]
    
    @pytest.mark.parametrize('transformer', TRANSFORMERS)
    def test_is_transformer(self, transformer):
        """Is the transformer a subclass of sklearn base classes?
        """
        assert isinstance(transformer, BaseEstimator)
        assert isinstance(transformer, TransformerMixin)
    
    @pytest.mark.parametrize('transformer', TRANSFORMERS)    
    def test_fit_transform(self, transformer, data):
        """Can the transformer be fit? Can it transform data?
        """
        transformer.fit(data)
        transformed = transformer.transform(data)
        fit_transformed = transformer.fit_transform(data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert isinstance(fit_transformed, pd.DataFrame)
        assert transformed.equals(fit_transformed)

    @pytest.mark.parametrize('transformer', TRANSFORMERS)        
    def test_pipelineable(self, transformer, data):
        """Is the transformer compatible with sklearn.pipeline.Pipeline?
        """
        pipe = Pipeline(steps=[('tran', transformer)])
        transformed = pipe.fit_transform(data)
        
        assert isinstance(transformed, pd.DataFrame)

    @pytest.mark.parametrize('transformer', TRANSFORMERS)
    def test_pickleable(self, transformer, data):
        """Can the transformer be pickled?
        """
        # save and load using temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = temp_dir + 'tran.pkl'
        
        before_pkl = transformer.fit_transform(data)
        
        with open(temp_path, 'wb') as tmp_f:
            pickle.dump(transformer, tmp_f)
            
        with open(temp_path, 'rb') as tmp_f:
            transformer = pickle.load(tmp_f)
            
        after_pkl = transformer.transform(data)
        
        assert before_pkl.equals(after_pkl)


class TestColumnTransformerDF():
    """Tests specific to tran.ColumnTransformerDF
    """
    
    @pytest.fixture(scope='class')
    def transformer(self):
        return tran.ColumnTransformerDF([('sc', StandardScaler(), ['num1'])],
                                         remainder='passthrough')
    
    def test_reconstruction(self, transformer, data):
        """Does the transformer preserve column ordering and dtypes?
        """
        transformed = transformer.fit_transform(data)
        
        assert np.all(data.columns, transformed.columns)
        assert data.dtypes.equals(transformed.dtypes)
        assert data.shape == transformed.shape

       
class TestUnivariateTransformer():
    """Test specific to tran.UnivariateTransformer
    """
    
    VALID_FUNCTIONS = [np.log, np.sqrt, functools.partial(np.power, 2)]
    
    @pytest.mark.parametrize('func', VALID_FUNCTIONS)
    def test_valid_functions(self, func, data):
        """Can the transformer apply different functions?
        """
        transformer = tran.UnivariateTransformer(variables=['num1', 'num2'],
                                                 func=func)
        transformed = transformer.fit_transform(data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert data.shape == transformed.shape
        
    def test_invalid(self, data):
        """Does the transformer raise correct (custom) error?
        """
        transformer = tran.UnivariateTransformer(variables=['num1', 'num2'],
                                                 func=math.log)
        
        with pytest.raises(InvalidInputError):
            transformer.fit_transform(data)
            
            
class TestBivariateTransformer():
    """Test specific to tran.BivariateTransformer
    """
    
    VALID_FUNCTIONS = [diff, add, product, leave_first]
    
    @pytest.mark.parametrize('func', VALID_FUNCTIONS)
    def test_valid_functions(self, func, data):
        """Can the transformer apply different functions?
        """
        transformer = tran.BivariateTransformer(variables=['num1'],
                                                reference_var='num2',
                                                func=func)
        transformed = transformer.fit_transform(data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert data.shape == transformed.shape
        
    def test_invalid(self, data):
        """Does the transformer raise correct (custom) error?
        """
        transformer = tran.UnivariateTransformer(variables=['num1', 'num2'],
                                                 func=math.log)
        
        with pytest.raises(InvalidInputError):
            transformer.fit_transform(data)
            
            
class TestFeatureDropper():
    """Tests specific to tran.FeatureDropper
    """
    
    @pytest.fixture(scope='class')
    def transformer(self):
        return tran.FeatureDropper(vars_to_drop=['num1', 'cat'])
    
    def test_drop(self, transformer, data):
        """Can the transformer drop the correct features?
        """
        transformed = transformer.transform(data)
        
        assert transformed.shape == (TEST_DATA_SIZE, 1)
        assert list(transformed.columns) == ['num2']
        
        
class TestRareLabelEncoder():
    """Tests specific to tran.RareLabelEncoder
    """
    
    @pytest.fixture(scope='class')
    def transformer(self):
        return tran.RareLabelEncoder(variables=['cat'])
    
    def test_encode(self, transformer, data):
        """Can the transformer correctly encode the rare label?
        """
        transformed = transformer.fit_transform(data)
        
        assert set(transformed['cat'].unique()) == {'a', 'b', 'c', 'rare'}
