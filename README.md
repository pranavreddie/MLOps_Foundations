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
|-- data/              # Data used in the pipelines
|-- notebooks/         # Jupyter notebooks for exploration
|-- scripts/           # Scripts for training and evaluation
|-- config/            # Configuration files
|-- tests/             # Unit and integration tests
|-- docker/            # Docker setup files
|-- .github/workflows/ # CI/CD workflows
|-- README.md          # Project overview
|-- requirements.txt   # Python dependencies
```

## Getting Started
### Prerequisites
1. Python (>=3.8)
2. Git
3. Docker (optional but recommended for containerization)
4. Basic understanding of machine learning and MLOps concepts
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
### Docker Setup (Optional)
Build and run the Docker container:
```docker build -t mlops-foundations .
docker run -it -p 8080:8080 mlops-foundations
```
---
## Key Features
1. **Pipeline Automation**: Automated workflows using CI/CD tools.
2. **Data Versioning**: Includes tools and best practices for data tracking.
3. **Model Training & Evaluation**: Scripts and notebooks to train and validate ML models.
4. **Containerization**: Docker support for consistent environments.
5. **Testing Framework**: Includes unit and integration tests.

### Additional Details on Models and Flask Application
#### Models Tried and Best Model
The project explored various machine learning models, including:
1. **Linear Regression**: Baseline model for comparison.
2. **Random Forest**: Provided significant improvements in accuracy and interpretability.
3. **Gradient Boosting**: Achieved the best performance with optimized hyperparameters.
The best model, Gradient Boosting, was saved as a serialized file (`model.pkl`) for deployment.
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
Navigate to the `scripts` directory and run the desired script. For example, to train a model:
python scripts/train_model.py --config config/model_config.yaml
### Jupyter Notebooks
Open and run the notebooks in the `notebooks/` directory for exploration and prototyping:
jupyter notebook
### Testing
Run the test suite to ensure functionality:
pytest tests/
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
For questions or support, please open an issue on the GitHub repository or contact the maintainer at
[pranavreddie@example.com](mailto:pranavreddie@example.com).

---
## Acknowledgments
Special thanks to contributors and the open-source community for their support and inspiration.

