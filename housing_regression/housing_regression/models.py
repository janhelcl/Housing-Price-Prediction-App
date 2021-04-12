"""
Registry of all available models
"""
import housing_regression.config.dev_config as dev_config
from housing_regression.pipelines.dev_pipeline import dev_pipeline


MODELS = {
        dev_config.NAME: {'config': dev_config,
                          'pipeline': dev_pipeline}
        }
