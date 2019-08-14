"""
Script to score new data with a persisted pipeline

TODO: extend to select one of multiple pipeliness
"""
import argparse
from typing import Dict, Any

import pandas as pd

import housing_regression.config.dev_config as conf
import housing_regression.processing.data_management as dm


#TODO: configurable
PIPELINE_PATH = 'housing_regression/trained_models/pipe.pkl'
_pipeline = dm.load_pipeline(PIPELINE_PATH)


def predict(input_data: Dict[str, Any]) -> dict:
    """Make prediction using persisted pipeline
    """
    data = pd.read_json(input_data)
    prediction = _pipeline.predict(data[conf.FEATURES])
    # np.ndarray is not JSON serializable
    return {'prediction': prediction.tolist()}

    
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('input_data', help='input data JSON')


if __name__ == '__main__':
    predict(parser.parse_args().input_data)
