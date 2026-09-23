# Credit Card Fraud Detection System

A machine learning-based web application for detecting potentially fraudulent credit card transactions using Python, Scikit-learn, and Flask.

## Project Overview

This project uses machine learning classification models to predict whether a credit card transaction is fraudulent.

The application provides a Flask-based web interface for:

- Training machine learning models
- Predicting fraudulent transactions
- Viewing data analysis
- Comparing model performance
- Visualizing results using interactive Plotly charts

## Machine Learning Models

The project uses the following classification models:

- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- HTML/CSS

## Data Processing

The application performs preprocessing on the transaction dataset, including:

- Loading data from CSV
- Cleaning column names
- Encoding categorical features
- Separating features and target variable
- Splitting data into training and testing sets
- Training multiple classification models

## Project Structure

```text
Credit_card_fraud_flask/
│
├── templates/
│   └── index.html
│
├── fraud_app.py
├── fraud_data.csv
├── requirements.txt
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore