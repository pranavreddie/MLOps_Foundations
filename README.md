# MLOPS Assignment-1
## Team Members
1. Amrita Chaudri (2023aa05132@wilp.bits-pilani.ac.in)
2. Arindam Sudhinkumar Ray (2023aa05722@wilp.bits-pilani.ac.in)
3. Bayana Narasimha Vara Pranav Reddy (2023aa05717@wilp.bits-pilani.ac.in)
4. Sudheesh K (2023aa05503@wilp.bits-pilani.ac.in)

## Project Overview

This project, hosted on GitHub at [MLOps_Foundations](https://github.com/pranavreddie/MLOps_Foundations/tree/main), aims to
provide foundational principles and practices for implementing Machine Learning Operations
(MLOps). It contains code and examples that demonstrate end-to-end machine learning workflows,
focusing on the automation and monitoring of machine learning pipelines.

---
## Table of Contents
1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Key Features](#key-features)
5. [Usage](#usage)
6. [Contributing](#contributing)
---
## Project Structure
The repository is organized as follows:

```
MLOps_Foundations/
|-- .dvc               # DVC details
|-- .github/workflows/ # CI/CD workflows
|-- data/              # Data used in the pipelines
|-- mlruns             # Contains Experiment tracking
|-- templates          # Include the front end for flask application
|-- app.py             # Flask Application
|-- main.py            # Overall Model Training
|-- README.md          # Project overview
|-- requirements.txt   # Python dependencies
|-- test_app.py        # Test case for app.py
|-- test_main.py       # Test case for main.py
```

## Getting Started
### Prerequisites
```
1. Python (>=3.8)
2. Git
3. MLFlow 
4. Docker (optional but recommended for containerization)
5. Basic understanding of machine learning and MLOps concepts
```
### Clone the Repository
```
git clone https://github.com/pranavreddie/MLOps_Foundations.git
cd MLOps_Foundations
```
---
## Installation
### Virtual Environment Setup
```
Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
### Install Dependencies
Install the required Python packages:
```
pip install -r requirements.txt
```
### MLFlow for Experiment Tracking:
```
python main.py
mlflow ui
```
### Docker Setup (Optional)
Build and run the Docker container:
```docker build -t mlops-foundations .
docker run -it -p 8080:8080 mlops-foundations
```
---
## Key Features
1. **Pipeline Automation**: Automated workflows using CI/CD tools to ensure consistent and reproducible model training and deployment processes.
2. **Data Versioning**: Tools for data tracking with DVC (Data Version Control), allowing easy versioning and management of datasets.
3. **Model Training & Evaluation**: Scripts are used to train and validate ML models.
4. **Containerization**: Docker support to ensure the consistency of the environment across different stages of the machine learning lifecycle.
5. **Testing Framework**: Includes unit and integration tests to ensure code reliability and correctness.
6. **MLflow for Experiment Tracking**: Integrated with MLflow to track experiments, log hyperparameters, and monitor model performance over time.

### Additional Details on Models and Flask Application
#### Models Tried and Best Model
The project explored multiple machine learning models to solve the problem. The following models were tested:
1. **Random Forest Classifier‎**: This baseline model provided a good initial performance and was used for comparison.
2. **Gradient Boosting**: Offered improvements in accuracy and model interpretability.
3. **Ada Boost Classifier**: While performing well, the model did not surpass Random Forest in terms of final performance.
   Best Model: Random Forest Classifier
   Random Forest achieved the best performance after training on the dataset. This model was selected for further fine-tuning, where hyperparameter optimization was performed to further improve its performance.
   Hyperparameter Tuning was carried out using tools like GridSearchCV or Optuna to find the best parameters for the Random Forest model.
   The best model, Random Forest Classifier, was saved as a serialized file (`model.pkl`) for deployment.
#### Flask Application
A Flask application was developed to serve the trained model. The key features of the Flask app
include:
- **Model Inference API**: Exposes an endpoint (`/predict`) to accept input data and return
  predictions.
- **User-Friendly Interface**: Simplifies testing and interaction with the deployed model.
  To run the Flask app:
  python scripts/run_flask_app.py
  The application is accessible locally at `http://localhost:5000`.
---
## Usage
### Running Scripts
python main.py
### Testing
Run the test suite to ensure functionality:
pytest test_main.py
pytest test_app.py
---
## Contributing
### How to Contribute
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and submit a pull request.
### Guidelines
- Follow PEP 8 standards for Python code.
- Write clear and concise commit messages.
- Include tests for any new features.
---
## Contact
For questions or support, please open an issue on the GitHub repository or contact the project team members.

---