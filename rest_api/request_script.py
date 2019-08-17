"""
Script to send requests to API endpoints
"""
import requests


URL = 'http://127.0.0.1:5000/'
VERSION_ENDPOINT = URL + 'version'


if __name__ == '__main__':
    version_response = requests.get(VERSION_ENDPOINT)
    print(version_response.text)
