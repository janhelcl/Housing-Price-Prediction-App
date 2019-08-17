"""
Application serving the housing regression models
"""
from flask import Flask


def create_app():
    """Housing regression application factory
    """
    app = Flask(__name__)
    # TODO: register blueprints
    
    return app
