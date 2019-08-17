"""
Functionality to manage data - datasets, persisted pipelines etc.

Will be extended in the future to handling multiple pipelines, logging etc.
"""
import logging

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


_logger = logging.getLogger(__name__)


def load_dataset(path: str) -> pd.DataFrame:
    """Loads csv data
    """
    _logger.info(f'loading data from {path}')
    return pd.read_csv(path)


def save_pipeline(pipe: Pipeline, path: str) -> None:
    """Save pipeline
    """
    _logger.info(f'saving pipeline to {path}')
    joblib.dump(pipe, path)


def load_pipeline(path: str) -> Pipeline:
    """Load a persisted pipeline
    """
    _logger.info(f'loading pipeline from {path}')
    trained_model = joblib.load(filename=path)
    return trained_model