"""
Tests the prediction function
"""
import json
import sys
sys.path.append('..')

from housing_regression.predict import predict
from housing_regression.processing.data_management import load_dataset


def test_single_predict():
    """Can the dev pipeline predict a previously unseen observation?
    """
    test_data = load_dataset('../housing_regression/data/test.csv')
    single_json = test_data.iloc[[0]].to_json(orient='records')
    scored = predict(single_json,
                     '../housing_regression/trained_models/dev_pipe.pkl')
    
    assert scored is not None
    assert isinstance(scored['prediction'][0], float)
    assert json.dumps(scored)
