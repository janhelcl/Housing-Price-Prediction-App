"""
Application serving the housing regression models
"""
import os

from flask import Flask

from api.blueprints.version_endpoint import version_endpoint


with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as ver_f:
    __version__ = ver_f.read().strip()
    
    
def create_app():
    """Housing regression application factory
    """
    app = Flask(__name__)
    
    app.register_blueprint(version_endpoint)
    
    return app
