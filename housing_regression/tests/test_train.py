"""
Test the train function
"""
import sys
import tempfile

sys.path.append("..")

import pytest
from sklearn.pipeline import Pipeline

from housing_regression.models import MODELS
from housing_regression.processing.data_management import load_pipeline
from housing_regression.train import train_pipeline

TRAIN_DATA = "housing_regression/data/train.csv"
MODEL_NAMES = MODELS.keys()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_train_pipeline(model_name):
    """Can the dev pipeline be trained and saved?"""
    # train the dev pipeline and save it to temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = temp_dir + "pipe.pkl"
    train_pipeline(TRAIN_DATA, model_name, temp_path)
    pipeline = load_pipeline(temp_path)

    assert isinstance(pipeline, Pipeline)
