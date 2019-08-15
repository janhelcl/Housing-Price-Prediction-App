"""

"""
from typing import Dict, Any

import pandas as pd

import config.dev_config as conf
import processing.data_management as dm


#TODO: configurable
#TODO: fix path
PIPELINE_PATH = 'C:/Users/jhelcl001/Desktop/housing_regression/Housing-Price-Prediction-App/housing_regression/trained_models/pipe.pkl'
_pipeline = dm.load_pipeline(PIPELINE_PATH)


def predict(input_data: Dict[str, Any]) -> dict:
    """Make prediction using persisted pipeline
    """
    data = pd.read_json(input_data)
    prediction = _pipeline.predict(data[conf.FEATURES])
    # np.ndarray is not JSON serializable
    return {'prediction': prediction.tolist()}
