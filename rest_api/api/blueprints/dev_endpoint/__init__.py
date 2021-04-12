"""
Endpoint serving the development model

This model was used during development of housing-regression package and is
not based on any sort of analysis. This endpoint can be used for testing of
both packages but should never be used for scoring.
"""
from flask import Blueprint, request, jsonify
from housing_regression.predict import predict


dev_endpoint = Blueprint('dev_endpoint', __name__)


@dev_endpoint.route('/predict/dev', methods=['POST'])
def make_prediction():
    """Returns predictions from the development model
    """
    input_data = request.get_json()
    return jsonify(predict(input_data, 'DevModel'))
