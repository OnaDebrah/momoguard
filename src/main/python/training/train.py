import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data.data_utils import save_model
from data.generate_data import generate_synthetic_data
from features.feature_engineering import engineer_features, get_feature_list
from train_model import train_model
from models.evaluate.evaluate_model import evaluate_model, find_optimal_threshold
from training.report import generate_conclusion_report
from utils.logger import logger
from visualisation.plots import FraudDetectionVisualizer


def main():
    try:
        # Debug timer
        start_time = time.time()

        # 1. Generate data (with smaller sample for debugging)
        logger.info("[1/6] Generating data...")
        df = generate_synthetic_data(n_samples=2000)  # Reduced from 5000 for debugging
        logger.info(f"Data generated. Shape: {df.shape}. Time: {time.time() - start_time:.2f}s")

        # 2. Feature engineering
        logger.info("[2/6] Engineering features...")
        enhanced_df = engineer_features(df)
        logger.info(f"Features engineered. Time: {time.time() - start_time:.2f}s")

        # 3. Model training
        logger.info("[3/6] Training model...")
        features = get_feature_list(enhanced_df)
        X = enhanced_df[features]
        y = enhanced_df['is_fraud']

        training_result = train_model(X, y)
        model = training_result['model']
        logger.info(f"Model trained. Time: {time.time() - start_time:.2f}s")

        # 4. Evaluation
        logger.info("[4/6] Evaluating model...")
        eval_result = evaluate_model(model, training_result['X_test'], training_result['y_test'])
        logger.info(f"Evaluation complete. Time: {time.time() - start_time:.2f}s")

        # 5. Threshold optimization
        logger.info("[5/6] Optimizing threshold...")
        best_threshold, best_f1 = find_optimal_threshold(
            training_result['y_test'],
            eval_result['y_prob']
        )
        logger.info(f"Optimal threshold found: {best_threshold:.2f}. Time: {time.time() - start_time:.2f}s")

        save_model({
            'model': training_result['model'],
            'threshold': best_threshold,
            'features': features
        }, 'fraud_detection_model.pkl')

        # Generate and print the report
        conclusion = generate_conclusion_report(eval_result, training_result, best_threshold, best_f1)
        from pprint import pprint
        pprint(conclusion)

        # Visualization
        logger.info("[6/6] Creating visualizations...")
        visualizer = FraudDetectionVisualizer()

        # 1. Exploratory Data Analysis Plots
        logger.info("\nCreating EDA plots...")
        visualizer.plot_amount_distribution(df)
        visualizer.plot_frequency_change_boxplot(df)
        visualizer.plot_correlation_matrix(df)
        visualizer.plot_feature_distributions(df)

        # 2. Feature Engineering Plots
        logger.info("\nCreating feature engineering plots...")
        visualizer.plot_risk_score_comparison(enhanced_df)
        visualizer.plot_feature_correlations(enhanced_df)

        # 3. Model Evaluation Plots
        logger.info("\nCreating model evaluation plots...")
        y_test = training_result['y_test']
        y_pred = training_result['y_pred']
        y_prob = training_result['y_prob']

        visualizer.plot_confusion_matrix(y_test, y_pred)
        visualizer.plot_roc_curve(y_test, y_prob)
        visualizer.plot_precision_recall_curve(y_test, y_prob)
        best_threshold, best_f1 = visualizer.plot_threshold_analysis(y_test, y_prob)

        feat_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        visualizer.plot_feature_importance(feat_importance)

        logger.info(f"All operations completed. Total time: {time.time() - start_time:.2f}s")

        return {
            'model': model,
            'best_threshold': best_threshold,
            'evaluation': eval_result
        }

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
    logger.info("Script executed successfully!")