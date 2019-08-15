"""
Functionality to train pipelines

# TODO: generalize to fit and persist a selected (not hardcoded) pipeline
"""
import housing_regression.config.dev_config as conf
import housing_regression.processing.data_management as dm
from housing_regression.pipelines.dev_pipeline import dev_pipeline

# TODO: configurable
TRAIN_FILE = 'data/train.csv'
SAVE_PATH = 'trained_models/pipe.pkl'


def train_pipeline() -> None:
    """Fit and persist the pipeline
    """
    data = dm.load_dataset(TRAIN_FILE)
    dev_pipeline.fit(data[conf.FEATURES], data[conf.LABEL])
    dm.save_pipeline(pipe=dev_pipeline, path=SAVE_PATH)
