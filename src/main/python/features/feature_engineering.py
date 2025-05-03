import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new engineered features for fraud detection

    Parameters:
    -----------
    df : pd.DataFrame
        Original transaction data with columns:
        - amount: Transaction amount
        - is_foreign_receiver: Binary foreign receiver indicator
        - num_recent_transactions: Count of recent transactions
        - avg_transaction_amount: Average transaction amount
        - transaction_frequency_change: Change in transaction frequency
        - is_new_receiver: Binary new receiver indicator
        - time_of_day_risk: Binary risky time indicator

    Returns:
    --------
    pd.DataFrame
        Data with additional engineered features:
        - amount_avg_ratio: Ratio of amount to average amount
        - combined_risk_score: Combined risk factors score
        - amount_risk: Risk score based on transaction amount
        - history_risk: Risk score based on transaction history
        - risk_score: Overall composite risk score
    """
    # Input validation
    required_columns = {
        'amount', 'is_foreign_receiver', 'num_recent_transactions',
        'avg_transaction_amount', 'transaction_frequency_change',
        'is_new_receiver', 'time_of_day_risk'
    }

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Create a copy to avoid modifying the original
    df_new = df.copy()

    # 1. Amount ratio feature
    df_new['amount_avg_ratio'] = create_amount_ratio_feature(
        df_new['amount'],
        df_new['avg_transaction_amount']
    )

    # 2. Combined risk factors
    df_new['combined_risk_score'] = calculate_combined_risk(
        df_new['is_foreign_receiver'],
        df_new['is_new_receiver'],
        df_new['time_of_day_risk'],
        df_new['transaction_frequency_change']
    )

    # 3. Amount risk feature
    df_new['amount_risk'] = calculate_amount_risk(df_new['amount'])

    # 4. History risk feature
    df_new['history_risk'] = calculate_history_risk(df_new['num_recent_transactions'])

    # 5. Composite risk score
    df_new['risk_score'] = calculate_composite_risk(
        df_new['combined_risk_score'],
        df_new['amount_risk'],
        df_new['history_risk']
    )

    return df_new

def get_feature_list(df: pd.DataFrame) -> List[str]:
    """
    Get the complete list of features including engineered ones

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing raw and/or engineered features

    Returns:
    --------
    List[str]
        Complete list of available features, prioritizing engineered ones
    """
    if 'feature_list' in df.attrs:
        return df.attrs['feature_list']
    return _get_feature_list(df)

# Internal implementation functions
def _get_feature_list(df: pd.DataFrame) -> List[str]:
    """Internal implementation of feature list generation"""
    engineered_features = [
        'amount_avg_ratio', 'combined_risk_score',
        'amount_risk', 'history_risk', 'risk_score'
    ]

    # Return engineered features first, then original ones
    return [f for f in engineered_features if f in df.columns] + \
        [f for f in df.columns if f not in engineered_features and f != 'is_fraud']

# Feature calculation helper functions
def create_amount_ratio_feature(amount: pd.Series, avg_amount: pd.Series) -> pd.Series:
    """Calculate ratio of transaction amount to average amount"""
    return amount / (avg_amount + 1)  # +1 to avoid division by zero

def calculate_combined_risk(
        is_foreign: pd.Series,
        is_new_receiver: pd.Series,
        time_risk: pd.Series,
        freq_change: pd.Series,
        weights: Dict[str, float] = {'foreign': 2.0, 'new_receiver': 1.5, 'time': 1.0, 'freq': 2.0}
) -> pd.Series:
    """Calculate combined risk score from individual factors"""
    return (
            is_foreign * weights['foreign'] +
            is_new_receiver * weights['new_receiver'] +
            time_risk * weights['time'] +
            (freq_change > 1) * weights['freq']
    )

def calculate_amount_risk(amount: pd.Series, scale_factor: float = 10.0) -> pd.Series:
    """Calculate risk score based on transaction amount (log-scaled)"""
    return np.log1p(amount) / scale_factor

def calculate_history_risk(
        num_transactions: pd.Series,
        decay_factor: float = 10.0
) -> pd.Series:
    """Calculate risk based on transaction history (exponential decay)"""
    return np.exp(-num_transactions / decay_factor)

def calculate_composite_risk(
        combined_score: pd.Series,
        amount_risk: pd.Series,
        history_risk: pd.Series,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> pd.Series:
    """Calculate final composite risk score"""
    return (
            combined_score * weights[0] +
            amount_risk * weights[1] +
            history_risk * weights[2]
    )

def preview_enhanced_data(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """Preview the enhanced dataframe with new features"""
    return df.head(n_rows)

# Example usage
# if __name__ == "__main__":
#     # Example: Generate synthetic data and apply feature engineering
#     from data.generate_data import generate_synthetic_data  # Assuming this is from the previous extraction
#
#     print("Generating synthetic data...")
#     transactions_df = generate_synthetic_data(n_samples=1000)
#
#     print("\nEngineering features...")
#     enhanced_df = engineer_features(transactions_df)
#
#     print("\nEnhanced data preview:")
#     print(preview_enhanced_data(enhanced_df))
#
#     print("\nNew features summary statistics:")
#     new_features = ['amount_avg_ratio', 'combined_risk_score',
#                     'amount_risk', 'history_risk', 'risk_score']
#     print(enhanced_df[new_features].describe())