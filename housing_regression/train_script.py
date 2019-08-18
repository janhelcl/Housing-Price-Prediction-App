"""
Script to fit and persist the pipeline
"""
import argparse

from housing_regression.train import train_pipeline


# defaults to dev pipeline
TRAIN_FILE = './housing_regression/data/train.csv'
MODEL = 'DevModel'


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--data', help='path to train data', default=TRAIN_FILE)
parser.add_argument('--model', help='name of registered model', default=MODEL)


if __name__ == '__main__':
    args = parser.parse_args()
    train_pipeline(data_path=args.data, model_name=args.model)
