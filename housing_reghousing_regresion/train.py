"""
Functionality to train pipelines

# TODO extend to mupltiple pipeline definitions
"""
import logging

import housing_regression.config.dev_config as conf
import housing_regression.processing.data_management as dm
from housing_regression.pipelines.dev_pipeline import dev_pipeline
from housing_regression import __version__


_logger = logging.getLogger(__name__)


def train_pipeline(data_path: str, pipeline_path: str) -> None:
    """Fit and persist the pipeline
    
    :param data_path: path to training dataset
    :param pipeline_path: where to save the pipeline
    """
    data = dm.load_dataset(data_path)
    _logger.info(f'Training pipeline version: {__version__}')
    dev_pipeline.fit(data[conf.FEATURES], data[conf.LABEL])
    dm.save_pipeline(pipe=dev_pipeline, path=pipeline_path)
