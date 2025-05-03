import pandas as pd
import joblib

def save_data(df, filepath):
    """Save DataFrame to file"""
    df.to_csv(filepath, index=False)

def load_data(filepath):
    """Load DataFrame from file"""
    return pd.read_csv(filepath)

def save_model(model, filepath):
    """Save trained model"""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load trained model"""
    return joblib.load(filepath)