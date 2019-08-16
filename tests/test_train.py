"""
Test the train function
"""
import tempfile
import sys
sys.path.append('..')

from sklearn.pipeline import Pipeline

from housing_regression.train import train_pipeline
from housing_regression.processing.data_management import load_pipeline


def test_train_pipeline():
    """Can the dev pipeline be trained and saved?
    """
    # train the dev pipeline and save it to temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = temp_dir + 'pipe.pkl'
    train_pipeline('../housing_regression/data/train.csv', temp_path)
    pipeline = load_pipeline(temp_path)
    
    assert isinstance(pipeline, Pipeline)