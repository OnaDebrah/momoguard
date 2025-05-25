import json
from datetime import datetime

import joblib
import pandas as pd

from utils.logger import logger

# Load the model package
model_pkg = joblib.load('fraud_detection_model.pkl')
model = model_pkg['model']
threshold = model_pkg['threshold']
required_features = model_pkg['features']

with open('sample_transaction.json', 'r') as f:
    sample_transactions = json.load(f)

def preprocess_transaction(tx):
    """Convert raw JSON transaction to model features"""
    tx_time = datetime.strptime(tx['timestamp'], '%Y-%m-%dT%H:%M:%S')

    features = {
        'amount': tx['amount'],
        'is_foreign_receiver': 0 if tx['receiver_phone'].startswith('+233') else 1,
        'num_recent_transactions': 5,  # Should come from your DB
        'avg_transaction_amount': 100.0,  # Should come from your DB
        'transaction_frequency_change': 0.2,  # Should be calculated
        'is_new_receiver': 0,  # 1 if first time with this receiver
        'time_of_day_risk': 1 if tx_time.hour < 6 else 0,  # Night=risky
        'amount_avg_ratio': tx['amount'] / 100.0,
        'combined_risk_score': 0,  # Calculate from other features
        'amount_risk': 0,  # Calculate from amount
        'history_risk': 0,  # Calculate from history
        'risk_score': 0  # Composite score
    }
    return pd.DataFrame([features])[model_pkg['features']]  # Ensure correct feature order

for tx in sample_transactions:
    features = preprocess_transaction(tx)
    fraud_prob = model.predict_proba(features)[0][1]
    tx['fraud_probability'] = float(fraud_prob)
    tx['is_fraud_predicted'] = bool(fraud_prob > threshold)

# 5. View results
logger.info(json.dumps(sample_transactions, indent=2))