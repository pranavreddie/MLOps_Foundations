from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load('best_tuned_model.pkl')

# Initialize the Flask application
app = Flask(__name__)

# Define the class labels
class_labels = ['setosa', 'versicolor', 'virginica']


# Route to display the HTML form
@app.route("/")
def home():
    # Return the home page with no prediction initially
    return render_template(
        'index.html',
        prediction=None,
        sepal_length=None,
        sepal_width=None,
        petal_length=None,
        petal_width=None
    )


# Route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    prediction = None  # Reset prediction to None when form is submitted again

    try:
        # Get the feature values from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Convert to a numpy array and reshape
        features = np.array(
            [sepal_length, sepal_width, petal_length, petal_width]
        ).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features)
        predicted_label = prediction[0]

        # Return the result with the prediction and the entered values
        return render_template(
            'index.html',
            prediction=predicted_label,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )

    except Exception as e:
        # Handle errors (e.g., invalid input) and display error message
        return render_template(
            'index.html',
            error=str(e),
            prediction=None,
            sepal_length=None,
            sepal_width=None,
            petal_length=None,
            petal_width=None
        )


# Run the Flask application
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080, debug=True)
