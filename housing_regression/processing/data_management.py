"""
Functionality to manage data - datasets, persisted pipelines etc.

Will be extended in the future to handling multiple pipelines, logging etc.
"""
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_dataset(path: str) -> pd.DataFrame:
    """Loads csv data
    """
    # TODO: logging
    return pd.read_csv(path)


def save_pipeline(pipe: Pipeline, path: str) -> None:
    """Save pipeline
    """
    # TODO: logging, managing other pipelines
    joblib.dump(pipe, path)


def load_pipeline(path: str) -> Pipeline:
    """Load a persisted pipeline
    """
    #TODO: logging
    trained_model = joblib.load(filename=path)
    return trained_model