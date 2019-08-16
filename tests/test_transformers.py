"""
Test the transformers
"""
import pickle
import tempfile
import sys
sys.path.append('..')

import pytest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import housing_regression.processing.transformers as tran


def diff(a, b):
    # dummy function to test tran.BivariateTransformer
    return a + b


class TestGeneralProperties():
    """Test common properties of all transformers
    """
    TEST_DATA_SIZE = 1000
    TRANSFORMERS = [
            tran.ColumnTransformerDF([('scaler', LabelEncoder(), ['cat'])]),
            tran.UnivariateTransformer(variables=['num2'], func=np.log),
            tran.BivariateTransformer(variables=['num1'],
                                      reference_var='num2',
                                      func=diff
                                      ),
            tran.FeatureDropper(vars_to_drop=['num1', 'num2']),
            tran.RareLabelEncoder(variables=['cat'])
    ]
    
    @pytest.fixture(scope='class')
    def data(self):
        data = {
        'num1': np.random.normal(size=self.TEST_DATA_SIZE),
        'num2': np.random.normal(size=self.TEST_DATA_SIZE),
        'cat': np.random.choice(['a', 'b', 'c'], size=self.TEST_DATA_SIZE)
        }
        data['cat'][0] = 'd' # add a rare category for tran.RareLabelEncoder
        return pd.DataFrame.from_dict(data)
    
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
        
        with open(temp_path) as tmp_f:
            pickle.dump(transformer, tmp_f)
            
        with open(temp_path) as tmp_f:
            transformer = pickle.load(tmp_f)
            
        after_pkl = transformer.transform(data)
        
        assert before_pkl.equals(after_pkl)
        