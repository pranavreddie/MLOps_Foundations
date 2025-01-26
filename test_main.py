import numpy as np
from app import model  # Assuming model is loaded in your Flask app


# Test the model loading
def test_model_loading():
    assert model is not None
    # Check if the model is loaded as an object
    assert isinstance(model, object)


# Test prediction function
def test_prediction():
    # Sample test data for prediction
    test_data = np.array([[5.1, 3.5, 1.4, 0.2]])

    # Make a prediction
    prediction = model.predict(test_data)

    assert prediction is not None
    assert prediction[0] in ['Setosa', 'Versicolor', 'Virginica']
