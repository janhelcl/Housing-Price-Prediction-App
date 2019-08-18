"""
Functionality to predict using persisted models
"""
import logging
from typing import Dict, Any

import pandas as pd

import housing_regression.processing.data_management as dm
from housing_regression.processing.validation import validate_inputs
from housing_regression.models import MODELS
from housing_regression import __version__


_logger = logging.getLogger(__name__)


def predict(input_data: Dict[str, Any], model_name: str) -> dict:
    """Make prediction using persisted pipeline
    
    :param input_data: data as JSON {"predictor_name": <predictor_value>, ...}
    :param model_name: name of a model registered in housing_regression.models
    """
    conf = MODELS[model_name]['config']
    
    pipeline = dm.load_pipeline(conf.PATH)
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
