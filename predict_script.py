"""
Script to score new data with a persisted pipeline

TODO: extend to select one of multiple pipeliness
"""
import argparse

from processing.predict import predict


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('input_data', help='input data JSON')


if __name__ == '__main__':
    predict(parser.parse_args().input_data)
