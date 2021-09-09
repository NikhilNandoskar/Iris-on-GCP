# Imports
from app import app
from flask import json

def test_api_call():
    data = {'sepal_length': 6.4, 'sepal_width': 3.2,
        'petal_length': 5.3, 'petal_width': 2.3,
        'model_name':'Support Vector Machine'}
    expected_response = {
        "model_name": "Support Vector Machine",
        "prediction": "Iris Virginica",
        "status": "Complete"
    }

    with app.test_client() as client:
        response = client.get('/test', query_string = data)
        assert response.status_code == 200
        assert json.loads(response.data) == expected_response

