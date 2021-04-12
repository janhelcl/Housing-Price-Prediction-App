"""
Script to send requests to API endpoints to see if they are alive
"""
import requests
import argparse


URL = 'http://127.0.0.1:5000/'
VERSION_ENDPOINT = URL + 'version'
DEV_ENDPOINT = URL + 'predict/dev'


SAMPLE_INPUT = open('../housing_regression/sample_input.json').read()

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--input_data', default=SAMPLE_INPUT)


if __name__ == '__main__':
    args = parser.parse_args()
    
    version_response = requests.get(VERSION_ENDPOINT)
    print(version_response.text)
    
    dev_response = requests.post(DEV_ENDPOINT, json=args.input_data)
    print(dev_response.text)
