import numpy as np
import pandas as pd
from typing import Dict, Any

def generate_synthetic_data(n_samples: int = 10000, fraud_ratio: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic mobile money transaction data with fraud indicators

    Parameters:
    -----------
    n_samples : int, optional
        Number of transactions to generate (default: 10000)
    fraud_ratio : float, optional
        Ratio of fraudulent transactions (default: 0.05)

    Returns:
    --------
    pd.DataFrame
        Synthetic transaction data with the following columns:
        - amount: Transaction amount (gamma distributed)
        - is_foreign_receiver: Binary indicator for foreign receiver
        - num_recent_transactions: Count of recent transactions (Poisson distributed)
        - avg_transaction_amount: Average transaction amount (gamma distributed)
        - transaction_frequency_change: Change in transaction frequency (normal distributed)
        - is_new_receiver: Binary indicator for new receiver
        - time_of_day_risk: Binary indicator for risky time of day (night)
        - is_fraud: Binary target variable indicating fraud
    """
    # Validate inputs
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not 0 <= fraud_ratio <= 1:
        raise ValueError("fraud_ratio must be between 0 and 1")

    # Calculate transaction counts
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud

    # Generate legitimate transactions
    legitimate_data: Dict[str, Any] = {
        'amount': np.random.gamma(shape=2, scale=100, size=n_legitimate),
        'is_foreign_receiver': np.random.choice([0, 1], size=n_legitimate, p=[0.95, 0.05]),
        'num_recent_transactions': np.random.poisson(lam=10, size=n_legitimate),
        'avg_transaction_amount': np.random.gamma(shape=2, scale=80, size=n_legitimate),
        'transaction_frequency_change': np.random.normal(loc=0, scale=0.3, size=n_legitimate),
        'is_new_receiver': np.random.choice([0, 1], size=n_legitimate, p=[0.7, 0.3]),
        'time_of_day_risk': np.random.choice([0, 1], size=n_legitimate, p=[0.85, 0.15]),
        'is_fraud': np.zeros(n_legitimate)
    }

    # Generate fraudulent transactions with distinct patterns
    fraud_data: Dict[str, Any] = {
        'amount': np.random.gamma(shape=5, scale=150, size=n_fraud),
        'is_foreign_receiver': np.random.choice([0, 1], size=n_fraud, p=[0.6, 0.4]),
        'num_recent_transactions': np.random.poisson(lam=3, size=n_fraud),
        'avg_transaction_amount': np.random.gamma(shape=1.5, scale=50, size=n_fraud),
        'transaction_frequency_change': np.random.normal(loc=1.5, scale=0.8, size=n_fraud),
        'is_new_receiver': np.random.choice([0, 1], size=n_fraud, p=[0.2, 0.8]),
        'time_of_day_risk': np.random.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
        'is_fraud': np.ones(n_fraud)
    }

    # Combine legitimate and fraudulent data
    for key in legitimate_data:
        legitimate_data[key] = np.concatenate([legitimate_data[key], fraud_data[key]])

    # Create DataFrame
    df = pd.DataFrame(legitimate_data)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

# def preview_data(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
#     """
#     Preview the first few rows of the generated data
#
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         The generated transaction data
#     n_rows : int, optional
#         Number of rows to display (default: 5)
#
#     Returns:
#     --------
#     pd.DataFrame
#         First n_rows of the data
#     """
#     return df.head(n_rows)
#
# # Example usage
# if __name__ == "__main__":
#     # Generate and preview data
#     transactions_df = generate_synthetic_data()
#     print("Generated transaction data preview:")
#     print(preview_data(transactions_df))
#
#     # Print basic statistics
#     print("\nBasic statistics:")
#     print(f"Total transactions: {len(transactions_df)}")
#     print(f"Fraudulent transactions: {transactions_df['is_fraud'].sum()} "
#           f"({transactions_df['is_fraud'].mean()*100:.2f}%)")