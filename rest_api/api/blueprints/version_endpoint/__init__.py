"""
Endpoint returning version of the application and models
"""
from flask import Blueprint, jsonify


version_endpoint = Blueprint('version_endpoint', __name__)


def get_api_version():
    """To avoid circular import
    """
    from api import __version__
    return  __version__


@version_endpoint.route('/version', methods=['GET'])
def version():
    """Returns version of the application
    """
    # TODO: add housing_regression version
    return jsonify({'api_version': get_api_version(),
                    'models_version': None
                    })
