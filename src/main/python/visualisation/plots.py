import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, \
    recall_score, f1_score, confusion_matrix


class FraudDetectionVisualizer:
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: tuple = (12, 8)):
        """Initialize visualizer with style settings"""
        self.set_style(style, figsize)

    def set_style(self, style: str, figsize: tuple):
        """Set visualization style"""
        plt.style.use(style)
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 12
        sns.set_palette("viridis")

    def plot_amount_distribution(self, df: pd.DataFrame, amount_col: str = 'amount',
                                 fraud_col: str = 'is_fraud', xlim: Optional[tuple] = (0, 1500)):
        """Plot distribution of transaction amounts by fraud status"""
        plt.figure(figsize=(12, 6))
        ax = sns.histplot(data=df, x=amount_col, hue=fraud_col, bins=50,
                          kde=True, element='step', common_norm=False)
        plt.title('Distribution of Transaction Amounts by Fraud Status')
        plt.xlabel('Transaction Amount (GHS)')
        plt.ylabel('Density')
        plt.legend(title='Fraud Status', labels=['Legitimate', 'Fraudulent'])
        if xlim:
            plt.xlim(xlim)
        plt.show()

    def plot_frequency_change_boxplot(self, df: pd.DataFrame, freq_col: str = 'transaction_frequency_change',
                                      fraud_col: str = 'is_fraud'):
        """Boxplot of transaction frequency change by fraud status"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=fraud_col, y=freq_col)
        plt.title('Transaction Frequency Change by Fraud Status')
        plt.xlabel('Fraud Status')
        plt.ylabel('Transaction Frequency Change')
        plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
        plt.show()

    def plot_correlation_matrix(self, df: pd.DataFrame, cmap: str = 'coolwarm'):
        """Plot correlation matrix of features"""
        plt.figure(figsize=(14, 10))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmin=-1, vmax=1,
                    fmt='.2f', center=0, square=True, linewidths=.5)
        plt.title('Correlation Matrix of Transaction Features')
        plt.tight_layout()
        plt.show()

    def plot_feature_distributions(self, df: pd.DataFrame, fraud_col: str = 'is_fraud',
                                   numerical_features: List[str] = None,
                                   categorical_features: List[str] = None):
        """
        Plot distributions of all features by fraud status

        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe containing features and fraud labels
        fraud_col : str
            Column name containing fraud labels (0/1)
        numerical_features : List[str]
            List of numerical feature names to plot
        categorical_features : List[str]
            List of categorical feature names to plot
        """
        if numerical_features is None:
            numerical_features = ['amount', 'num_recent_transactions',
                                  'avg_transaction_amount', 'transaction_frequency_change']

        if categorical_features is None:
            categorical_features = ['is_foreign_receiver', 'is_new_receiver', 'time_of_day_risk']

        # Calculate required grid size
        total_features = len(numerical_features) + len(categorical_features)
        nrows = (total_features + 2) // 3  # Ensure enough rows for all features
        fig, axes = plt.subplots(nrows, 3, figsize=(18, 6 * nrows))
        axes = axes.flatten()

        # Plot numerical features
        for i, feature in enumerate(numerical_features):
            sns.histplot(data=df, x=feature, hue=fraud_col, kde=True,
                         element='step', ax=axes[i], common_norm=False)
            axes[i].set_title(f'Distribution of {feature.replace("_", " ").title()}')
            axes[i].legend(['Legitimate', 'Fraudulent'])

        # Plot categorical features
        for i, feature in enumerate(categorical_features):
            idx = i + len(numerical_features)
            if idx < len(axes):  # Safety check
                sns.countplot(data=df, x=feature, hue=fraud_col, ax=axes[idx])
                axes[idx].set_title(f'Count of {feature.replace("_", " ").title()}')
                axes[idx].legend(['Legitimate', 'Fraudulent'])

        # Hide any unused axes
        for j in range(len(numerical_features) + len(categorical_features), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_risk_score_comparison(self, df: pd.DataFrame, score_col: str = 'risk_score',
                                   fraud_col: str = 'is_fraud'):
        """Boxplot comparing risk scores between fraud and non-fraud"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=fraud_col, y=score_col)
        plt.title('Risk Score by Fraud Status')
        plt.xlabel('Fraud Status')
        plt.ylabel('Risk Score')
        plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
        plt.show()

    def plot_feature_correlations(self, df: pd.DataFrame, target_col: str = 'is_fraud',
                                  features: List[str] = None):
        """Horizontal bar plot of feature correlations with target"""
        if features is None:
            features = ['amount_avg_ratio', 'combined_risk_score',
                        'amount_risk', 'history_risk', 'risk_score', 'is_fraud']

        corr = df[features].corr()[target_col].sort_values()
        plt.figure(figsize=(10, 6))
        corr.drop(target_col).plot(kind='barh')
        plt.title('Correlation of Engineered Features with Fraud')
        plt.xlabel('Correlation Coefficient')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Plot confusion matrix with annotations"""
        if labels is None:
            labels = ['Legitimate', 'Fraudulent']

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def plot_roc_curve(self, y_true, y_prob):
        """Plot ROC curve with AUC score"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_prob):
        """Plot precision-recall curve with average precision score"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'Precision-Recall (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def plot_threshold_analysis(self, y_true, y_prob, thresholds=None):
        """Plot precision, recall, and F1 across decision thresholds"""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 17)

        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred))

        plt.figure(figsize=(12, 6))

        # Plot precision and recall
        plt.plot(thresholds, precisions, 'b--', label='Precision')
        plt.plot(thresholds, recalls, 'g-', label='Recall')

        # Find and mark best F1 threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        plt.axvline(x=best_threshold, color='r', linestyle=':',
                    label=f'Best F1 Threshold ({best_threshold:.2f}, F1={best_f1:.2f})')

        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall and F1 Score vs. Decision Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()

        return best_threshold, best_f1

    def plot_feature_importance(self, feature_importance: pd.DataFrame):
        """Plot feature importance from model"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()


# Example usage
# if __name__ == "__main__":
#     # Generate sample data (assuming these functions exist)
#     from data.generate_data import generate_synthetic_data
#     from features.feature_engineering import engineer_features
#     from sklearn.model_selection import train_test_split
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.metrics import (confusion_matrix, roc_curve, auc,
#                                  precision_recall_curve, average_precision_score,
#                                  precision_score, recall_score, f1_score)
#
#     # Create visualizer instance
#     visualizer = FraudDetectionVisualizer()
#
#     # Generate and prepare data
#     print("Generating and preparing data...")
#     transactions = generate_synthetic_data(n_samples=5000)
#     enhanced_data = engineer_features(transactions)
#
#     # 1. Exploratory Data Analysis Plots
#     print("\nCreating EDA plots...")
#     visualizer.plot_amount_distribution(transactions)
#     visualizer.plot_frequency_change_boxplot(transactions)
#     visualizer.plot_correlation_matrix(transactions)
#     visualizer.plot_feature_distributions(transactions)
#
#     # 2. Feature Engineering Plots
#     print("\nCreating feature engineering plots...")
#     visualizer.plot_risk_score_comparison(enhanced_data)
#     visualizer.plot_feature_correlations(enhanced_data)
#
#     # 3. Model Evaluation Plots (example with simple model)
#     print("\nCreating model evaluation plots...")
#     X = enhanced_data.drop('is_fraud', axis=1)
#     y = enhanced_data['is_fraud']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]
#
#     visualizer.plot_confusion_matrix(y_test, y_pred)
#     visualizer.plot_roc_curve(y_test, y_prob)
#     visualizer.plot_precision_recall_curve(y_test, y_prob)
#     best_threshold, best_f1 = visualizer.plot_threshold_analysis(y_test, y_prob)
#
#     # Feature importance
#     feat_importance = pd.DataFrame({
#         'Feature': X.columns,
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False)
#
#     visualizer.plot_feature_importance(feat_importance)
