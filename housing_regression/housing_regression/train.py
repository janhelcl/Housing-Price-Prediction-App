"""
Functionality to train registered models
"""
import logging

import housing_regression.config.global_config as global_conf
import housing_regression.processing.data_management as dm
from housing_regression.models import MODELS
from housing_regression import __version__


_logger = logging.getLogger(__name__)


def train_pipeline(data_path: str, model_name: str, save_path=None) -> None:
    """Fit and persist the pipeline
    
    :param data_path: path to training dataset
    :param model_name: name of a model registered in housing_regression.models
    :param save_path: where to save the pipeline
    """
    data = dm.load_dataset(data_path)
    _logger.info(f'Training pipeline: {model_name}, version: {__version__}')
    
    pipeline = MODELS[model_name]['pipeline']
    conf = MODELS[model_name]['config']
    
    pipeline.fit(data[conf.FEATURES], data[global_conf.LABEL])
    if not save_path:
        save_path = conf.PATH
    dm.save_pipeline(pipe=pipeline, path=save_path)
