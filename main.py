import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset from CSV
df = pd.read_csv('data/iris.csv')

# Assuming the target column is 'variety' and the rest are features
X = df.drop(columns=['variety'])  # Features (exclude target column)
y = df['variety']  # Target column

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to DataFrame for input signature
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# Define the models
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, random_state=42, algorithm='SAMME'
    )
}

# Start an MLflow experiment
mlflow.set_experiment("Model Experimentation")

best_model = None
best_accuracy = 0
best_model_name = ""

# Step 1: Train multiple models and log their metrics
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)

        # Log model parameters
        mlflow.log_param("model_type", model_name)
        if hasattr(model, "n_estimators"):
            mlflow.log_param(
                "n_estimators", model.n_estimators
            )
        if hasattr(model, "max_depth"):
            mlflow.log_param(
                "max_depth", model.max_depth
            )
        if hasattr(model, "learning_rate"):
            mlflow.log_param(
                "learning_rate", getattr(model, "learning_rate", None)
            )

        # Log the model
        input_example = X_test_df.iloc[0:1]  # Example input row for logging
        signature = mlflow.models.infer_signature(
            X_train_df, model.predict(X_train)
        )
        mlflow.sklearn.log_model(
            model, "model", input_example=input_example, signature=signature
        )

        # Print results for reference
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Logged to MLflow\n")

        # Track the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

# After selecting the best model, perform hyperparameter tuning on it (Step 2)
if best_model_name:
    model_name = best_model_name+"_tuned_model"
    mlflow.start_run(run_name=model_name)
    print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")

    # Hyperparameter Grid for GridSearchCV (tuning for the best model)
    param_grid = {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        },
        "GradientBoosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1, 0.2]
        },
        "AdaBoost": {
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0]
        }
    }

    # Step 2: Perform GridSearchCV for hyperparameter tuning
    # only on the best model
    if best_model_name == "RandomForest":
        grid_search = GridSearchCV(
            estimator=best_model,
            param_grid=param_grid["RandomForest"],
            cv=5
        )
    elif best_model_name == "GradientBoosting":
        grid_search = GridSearchCV(
            estimator=best_model,
            param_grid=param_grid["GradientBoosting"],
            cv=5
        )
    elif best_model_name == "AdaBoost":
        grid_search = GridSearchCV(
            estimator=best_model,
            param_grid=param_grid["AdaBoost"],
            cv=5
        )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    tuned_model = grid_search.best_estimator_

    # Log tuned model parameters
    mlflow.log_params(best_params)

    # Re-evaluate the tuned model
    predictions = tuned_model.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("tuned_accuracy", tuned_accuracy)

    # Log the tuned model
    signature = mlflow.models.infer_signature(
        X_train_df, tuned_model.predict(X_train)
    )
    mlflow.sklearn.log_model(
        tuned_model, "tuned_model",
        input_example=X_test_df.iloc[0:1],
        signature=signature
    )

    print(f"Tuned model: {best_model_name}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Tuned Accuracy: {tuned_accuracy:.4f}")

    # Save the best tuned model for future use in M3 (packaging)
    joblib.dump(tuned_model, "best_tuned_model.pkl")
