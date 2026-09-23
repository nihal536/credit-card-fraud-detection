# Stock Analysis Dashboard

## Running Locally

To run this app on your local machine:

### Option 1: Simple Local Run
```bash
streamlit run app.py --server.address localhost --server.port 8501
```
Then open your browser to: `http://localhost:8501`

### Option 2: Using the Python Runner
```bash
python run_local.py
```

### Option 3: Default Streamlit (if you prefer default port)
```bash
streamlit run app.py
```
This will run on `http://localhost:8501` by default.

## For Replit Deployment
The app is configured to run on `0.0.0.0:5000` for deployment, but for local development use `localhost:8501`.

## Features
- Real-time stock data from Yahoo Finance
- Interactive charts (candlestick, line, area)
- Financial summary tables
- CSV export functionality
- Multiple time period analysis