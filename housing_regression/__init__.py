import os

from housing_regression.config.logging_config import get_logger


logger = get_logger(__name__)


with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as ver_f:
    __version__ = ver_f.read().strip()
