"""
Script to score new data with the pipeline

TODO: extend to select one of multiple pipelines
"""
import pandas as pd

import config.dev_config as conf
import processing.data_management as dm


#TODO: configurable
PIPELINE_PATH = 'trained_models/pipe.pkl'
_pipeline = dm.load_pipeline(PIPELINE_PATH)


def predict(input_data) -> dict:
    """Make prediction using persisted pipeline
    """
    data = pd.read_json(input_data)
    prediction = _pipeline.predict(data[conf.FEATURES])
    return {'prediction': prediction}