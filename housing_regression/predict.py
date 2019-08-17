"""
Functionality to predict using persisted

# TODO extend to mupltiple pipeline definitions
"""
import logging
from typing import Dict, Any

import pandas as pd

import housing_regression.config.dev_config as conf
import housing_regression.processing.data_management as dm
from housing_regression.processing.validation import validate_inputs
from housing_regression import __version__


_logger = logging.getLogger(__name__)


def predict(input_data: Dict[str, Any], pipeline_path: str) -> dict:
    """Make prediction using persisted pipeline
    
    :param input_data: data as JSON {"predictor_name": <predictor_value>, ...}
    :param pipeline_path: path to the saved pipeline
    """
    pipeline = dm.load_pipeline(pipeline_path)
    data = pd.read_json(input_data)
    validated = validate_inputs(data)
    
    prediction_array = pipeline.predict(validated[conf.FEATURES])
    # np.ndarray is not JSON serializable
    prediction = prediction_array.tolist()
    
    _logger.info(
        f'Made predictions with model version: {__version__} '
        f'Inputs: {validated} '
        f'Predictions: {prediction}')
        
    return {'prediction': prediction,
            'version': __version__}
