"""
Tests the prediction function
"""
import json
import sys
sys.path.append('..')

from housing_regression.predict import predict
from housing_regression.processing.data_management import load_dataset


TEST_DATA = '../housing_regression/data/test.csv'
DEV_PIPELINE = '../housing_regression/trained_models/dev_pipe.pkl'


def test_single_predict():
    """Can the dev pipeline predict a previously unseen observation?
    """
    test_data = load_dataset(TEST_DATA)
    single_json = test_data.iloc[[0]].to_json(orient='records')
    scored = predict(single_json, DEV_PIPELINE)
    
    assert scored is not None
    assert isinstance(scored['prediction'][0], float)
    assert json.dumps(scored)
    
    
def test_multiple_predict():
    """Can the dev pipeline predict multiple unseen observations?
    """
    test_data = load_dataset(TEST_DATA)
    single_json = test_data.to_json(orient='records')
    scored = predict(single_json, DEV_PIPELINE)
    
    assert scored is not None
    assert isinstance(scored['prediction'], list)
    assert len(scored['prediction']) <= len(test_data)
    assert json.dumps(scored)
