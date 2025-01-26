import pytest
from app import app  # Assuming app.py is your Flask application file


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_homepage(client):
    # Test the homepage loads correctly
    response = client.get('/')
    assert response.status_code == 200


def test_predict_valid_input(client):
    # Test valid POST request for prediction
    data = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Prediction:' in response.data


def test_predict_invalid_input(client):
    # Test invalid POST request (e.g., missing data)
    data = {
        'sepal_length': 'invalid',  # Invalid input
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Error:' in response.data


def test_home_page_form_values(client):
    # Test that input values are correctly populated after prediction
    data = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    response = client.post('/predict', data=data)
    assert b'Prediction:' in response.data
    assert b'5.1' in response.data
    assert b'3.5' in response.data
    assert b'1.4' in response.data
    assert b'0.2' in response.data
