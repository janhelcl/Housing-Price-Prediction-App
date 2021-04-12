"""
Endpoint returning version of the application and models
"""
from flask import Blueprint, jsonify
import housing_regression as hr


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
    return jsonify({'api_version': get_api_version(),
                    'models_version': hr.__version__
                    })
