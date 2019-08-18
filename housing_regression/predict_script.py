"""
Script to score new data with a persisted pipeline
"""
import argparse

from housing_regression.predict import predict


# defaults to dev pipeline
MODEL = 'DevModel'


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('input_data', help='input data JSON')
parser.add_argument('--model', 
                    help='name of registered model',
                    default=MODEL)


if __name__ == '__main__':
    args = parser.parse_args()
    predict(input_data=args.input_data, model_name=args.model)
