import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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


# Test accuracy with known test set
def test_accuracy():
    # Load dataset
    df = pd.read_csv('data/iris.csv')  # Update the path to your dataset
    X = df.drop(columns=['variety'])
    y = df['variety']

    # Evaluate model accuracy on the test set
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    # Assert that the accuracy is greater than or equal to 90%
    assert accuracy >= 0.9
