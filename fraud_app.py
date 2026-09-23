from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

class FraudDetectionModel:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the fraud data"""
        try:
            # Read CSV with semicolon separator
            df = pd.read_csv(filepath, sep=';')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle categorical variables
            categorical_columns = ['is_declined', 'foreign_transaction', 'high_risk_countries']
            
            for col in categorical_columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
            
            # Convert target variable
            if 'is_fradulent' in df.columns:
                target_encoder = LabelEncoder()
                df['is_fradulent'] = target_encoder.fit_transform(df['is_fradulent'])
                self.encoders['target'] = target_encoder
            
            # Select feature columns (excluding merchant_id and target)
            feature_cols = [col for col in df.columns if col not in ['merchan_id', 'is_fradulent']]
            self.feature_columns = feature_cols
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def train_models(self, df):
        """Train multiple ML models"""
        try:
            X = df[self.feature_columns]
            y = df['is_fradulent']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Define models
            models_to_train = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(probability=True, random_state=42)
            }
            
            # Train models and store results
            results = {}
            for name, model in models_to_train.items():
                if name == 'Random Forest':
                    # Random Forest doesn't need scaling
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # Other models benefit from scaling
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                self.models[name] = model
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                    'probabilities': y_pred_proba.tolist() if hasattr(y_pred_proba, 'tolist') else list(y_pred_proba),
                    'actual': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
                }
            
            self.is_trained = True
            return results, X_test, y_test
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            return None, None, None
    
    def predict_single(self, features, model_name='Random Forest'):
        """Make prediction for a single transaction"""
        try:
            if not self.is_trained:
                return None
            
            model = self.models.get(model_name)
            if not model:
                return None
            
            # Create feature array
            feature_array = np.array([features])
            
            # Scale if needed
            if model_name != 'Random Forest':
                feature_array = self.scaler.transform(feature_array)
            
            prediction = model.predict(feature_array)[0]
            probability = model.predict_proba(feature_array)[0][1]
            
            # Convert back to original labels
            prediction_label = self.encoders['target'].inverse_transform([prediction])[0]
            
            return {
                'prediction': prediction_label,
                'probability': float(probability),
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None

# Initialize model
fraud_model = FraudDetectionModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train the fraud detection models"""
    try:
        # Load and preprocess data
        df = fraud_model.load_and_preprocess_data('fraud_data.csv')
        if df is None:
            return jsonify({'error': 'Failed to load data'}), 400
        
        # Train models
        results, X_test, y_test = fraud_model.train_models(df)
        if results is None:
            return jsonify({'error': 'Failed to train models'}), 400
        
        # Create visualizations
        charts = create_training_charts(df, results)
        
        return jsonify({
            'success': True,
            'results': results,
            'charts': charts,
            'data_info': {
                'total_records': len(df),
                'fraud_percentage': (df['is_fradulent'].sum() / len(df)) * 100,
                'features': fraud_model.feature_columns
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make fraud prediction for new transaction"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        if not fraud_model.is_trained:
            return jsonify({'error': 'Models not trained yet. Please train models first.'}), 400
        
        model_name = data.get('model', 'Random Forest')
        
        # Extract features in the correct order
        features = []
        for col in fraud_model.feature_columns:
            value = data.get(col, 0)
            # Convert categorical values if needed
            if col == 'is_declined':
                value = 1 if str(value).lower() == 'yes' else 0
            elif col == 'foreign_transaction':
                value = 1 if str(value).lower() == 'yes' else 0
            elif col == 'high_risk_countries':
                value = 1 if str(value).lower() == 'yes' else 0
            features.append(float(value))
        
        # Make prediction
        result = fraud_model.predict_single(features, model_name)
        if result is None:
            return jsonify({'error': 'Failed to make prediction. Please ensure models are trained.'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data_analysis')
def data_analysis():
    """Get data analysis and visualizations"""
    try:
        df = fraud_model.load_and_preprocess_data('fraud_data.csv')
        if df is None:
            return jsonify({'error': 'Failed to load data'}), 400
        
        # Create analysis charts
        charts = create_analysis_charts(df)
        
        # Basic statistics
        stats = {
            'total_transactions': len(df),
            'fraud_count': int(df['is_fradulent'].sum()),
            'legitimate_count': int(len(df) - df['is_fradulent'].sum()),
            'fraud_rate': float((df['is_fradulent'].sum() / len(df)) * 100),
            'avg_transaction_amount': float(df['transaction_amount'].mean()),
            'median_transaction_amount': float(df['transaction_amount'].median())
        }
        
        return jsonify({
            'stats': stats,
            'charts': charts
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_training_charts(df, results):
    """Create charts for training results"""
    charts = {}
    
    # Model accuracy comparison
    accuracies = {model: result['accuracy'] for model, result in results.items()}
    
    fig_accuracy = go.Figure(data=[
        go.Bar(x=list(accuracies.keys()), y=list(accuracies.values()))
    ])
    fig_accuracy.update_layout(
        title='Model Accuracy Comparison',
        xaxis_title='Model',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1])
    )
    charts['accuracy'] = json.dumps(fig_accuracy, cls=PlotlyJSONEncoder)
    
    return charts

def create_analysis_charts(df):
    """Create data analysis charts"""
    charts = {}
    
    # Fraud distribution
    fraud_counts = df['is_fradulent'].value_counts()
    labels = ['Legitimate', 'Fraudulent']
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=fraud_counts.values,
        hole=0.3
    )])
    fig_pie.update_layout(title='Transaction Distribution: Fraud vs Legitimate')
    charts['fraud_distribution'] = json.dumps(fig_pie, cls=PlotlyJSONEncoder)
    
    # Transaction amount distribution
    fig_hist = go.Figure()
    for label, value in zip(['Legitimate', 'Fraudulent'], [0, 1]):
        data = df[df['is_fradulent'] == value]['transaction_amount']
        fig_hist.add_trace(go.Histogram(
            x=data,
            name=label,
            opacity=0.7,
            nbinsx=50
        ))
    
    fig_hist.update_layout(
        title='Transaction Amount Distribution by Fraud Status',
        xaxis_title='Transaction Amount',
        yaxis_title='Frequency',
        barmode='overlay'
    )
    charts['amount_distribution'] = json.dumps(fig_hist, cls=PlotlyJSONEncoder)
    
    # Foreign transaction analysis
    foreign_fraud = df.groupby(['foreign_transaction', 'is_fradulent']).size().unstack(fill_value=0)
    
    fig_bar = go.Figure(data=[
        go.Bar(name='Legitimate', x=['Domestic', 'Foreign'], y=foreign_fraud[0]),
        go.Bar(name='Fraudulent', x=['Domestic', 'Foreign'], y=foreign_fraud[1])
    ])
    fig_bar.update_layout(
        title='Fraud Distribution: Domestic vs Foreign Transactions',
        barmode='group'
    )
    charts['foreign_analysis'] = json.dumps(fig_bar, cls=PlotlyJSONEncoder)
    
    return charts

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)