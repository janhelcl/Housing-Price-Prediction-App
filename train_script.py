"""
Script to fit the pipeline

# TODO: generalize to fit and persist a selected (not hardcoded) pipeline
"""
import argparse

from housing_regression.train import train_pipeline


# defaults to dev pipeline
TRAIN_FILE = './housing_regression/data/train.csv'
SAVE_PATH = './housing_regression/trained_models/dev_pipe.pkl'


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--train', help='path to train file', default=TRAIN_FILE)
parser.add_argument('--save', help='where to save result', default=SAVE_PATH)


if __name__ == '__main__':
    args = parser.parse_args()
    train_pipeline(data_path=args.train, pipeline_path=args.save)
