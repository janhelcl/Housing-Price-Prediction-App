"""
Script to score new data with a persisted pipeline

TODO: extend to select one of multiple pipeliness
"""
import argparse

from housing_regression.predict import predict


# defaults to dev pipeline
PIPELINE_PATH = './housing_regression/trained_models/dev_pipe.pkl'


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('input_data', help='input data JSON')
parser.add_argument('--pipe', 
                    help='path to a persisted pipeline',
                    default=PIPELINE_PATH)


if __name__ == '__main__':
    args = parser.parse_args()
    predict(input_data=args.input_data, pipeline_path=args.pipe)
