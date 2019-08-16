"""
Functionality to predict using persisted

# TODO extend to mupltiple pipeline definitions
"""
from typing import Dict, Any

import pandas as pd

import housing_regression.config.dev_config as conf
import housing_regression.processing.data_management as dm


def predict(input_data: Dict[str, Any], pipeline_path: str) -> dict:
    """Make prediction using persisted pipeline
    
    :param input_data: data as JSON {"predictor_name": <predictor_value>, ...}
    :param pipeline_path: path to the saved pipeline
    """
    pipeline = dm.load_pipeline(pipeline_path)
    data = pd.read_json(input_data)
    prediction = pipeline.predict(data[conf.FEATURES])
    # np.ndarray is not JSON serializable
    return {'prediction': prediction.tolist()}
