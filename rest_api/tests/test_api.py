"""
Testing the API
"""
import sys
sys.path.append('..')

import pytest
from flask import json

import api
import housing_regression as hr


@pytest.fixture(scope='module')
def app():
    app = api.create_app()
    with app.app_context():
        yield app
        
        
@pytest.fixture(scope='module')
def client(app):
    with app.test_client() as client:
        yield client
        
        
def test_version_endpoint(client):
    """Does the version endpoint return correct versions?
    """
    response = client.get('/version')
    data = json.loads(response.data)
    
    assert response.status_code == 200
    assert data['api_version'] == api.__version__
    assert data['models_version'] == hr.__version__
