"""
Tests the prediction function
"""
import json
import sys

sys.path.append("..")

import pytest

from housing_regression.models import MODELS
from housing_regression.predict import predict
from housing_regression.processing.data_management import load_dataset

TEST_DATA = "housing_regression/data/test.csv"
MODEL_NAMES = MODELS.keys()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_single_predict(model_name):
    """Can the models predict a previously unseen observation?"""
    test_data = load_dataset(TEST_DATA)
    single_json = test_data.iloc[[0]].to_json(orient="records")
    scored = predict(single_json, model_name)

    assert scored is not None
    assert isinstance(scored["prediction"][0], float)
    assert json.dumps(scored)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_multiple_predict(model_name):
    """Can the models predict multiple unseen observations?"""
    test_data = load_dataset(TEST_DATA)
    single_json = test_data.to_json(orient="records")
    scored = predict(single_json, model_name)

    assert scored is not None
    assert isinstance(scored["prediction"], list)
    assert len(scored["prediction"]) <= len(test_data)
    assert json.dumps(scored)
