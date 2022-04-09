"""
Application serving the housing regression models
"""
import os

from flask import Flask

from rest_api.api.blueprints.version_endpoint import version_endpoint
from rest_api.api.blueprints.dev_endpoint import dev_endpoint


with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as ver_f:
    __version__ = ver_f.read().strip()
    
    
def create_app():
    """Housing regression application factory
    """
    app = Flask(__name__)
    
    app.register_blueprint(version_endpoint)
    app.register_blueprint(dev_endpoint)
    
    return app