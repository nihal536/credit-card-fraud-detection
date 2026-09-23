# Credit Card Fraud Detection System

A machine learning-based web application for detecting potentially fraudulent credit card transactions using Python, Scikit-learn, and Flask.

## Project Overview

This project applies machine learning classification techniques to predict whether a credit card transaction is fraudulent.

The Flask web application provides functionality for:

- Training multiple machine learning models
- Predicting individual transactions
- Viewing transaction data analysis
- Comparing model accuracy
- Visualizing analysis and training results using Plotly

## Machine Learning Models

The project uses three classification models:

- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)

## Data Preprocessing

The application performs the following preprocessing steps:

- Loads transaction data from a CSV file
- Cleans column names
- Encodes categorical variables using LabelEncoder
- Separates features and target variable
- Splits the dataset into training and testing sets
- Applies StandardScaler to Logistic Regression and SVM
- Trains the classification models

## Prediction

The application can make a prediction for an individual transaction.

For a selected model, the application returns:

- Predicted transaction class
- Prediction probability
- Risk level: Low, Medium, or High

## Data Analysis

The application provides interactive visualizations including:

- Fraudulent vs. legitimate transaction distribution
- Transaction amount distribution by fraud status
- Domestic vs. foreign transaction analysis
- Model accuracy comparison

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- HTML/CSS

## Project Structure

```text
Credit_card_fraud_flask/
│
├── templates/
│   └── index.html
│
├── screenshots/
│   └── dashboard.png
│
├── fraud_app.py
├── fraud_data.csv
├── requirements.txt
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore