def generate_conclusion_report(eval_result, training_result, best_threshold, best_f1):
    """Generate a structured conclusion from evaluation results"""
    try:
        # Get metrics at optimal threshold
        optimal_eval = eval_result  # Using the pre-computed evaluation

        # Extract classification report safely
        class_report = optimal_eval['classification_report']

        # Determine the fraud class key dynamically
        fraud_key = None

        # Case 1: Standard binary classification (0/1)
        if isinstance(class_report, dict) and '1' in class_report:
            fraud_key = '1'
        # Case 2: Boolean labels (True/False)
        elif isinstance(class_report, dict) and 'True' in class_report:
            fraud_key = 'True'
        # Case 3: String labels
        elif isinstance(class_report, dict) and len(class_report) >= 3:
            # Assuming the last key is the positive class
            possible_keys = [k for k in class_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            if len(possible_keys) >= 2:
                fraud_key = possible_keys[1]  # Second class is typically positive
        # Case 4: When classification_report is a string (convert to dict)
        elif isinstance(class_report, str):
            from sklearn.metrics import classification_report
            class_report = classification_report(
                optimal_eval['y_true'],
                optimal_eval['y_pred'],
                output_dict=True
            )
            return generate_conclusion_report({'classification_report': class_report}, training_result, best_threshold, best_f1)

        if fraud_key is None:
            raise ValueError("Could not identify fraud class in classification report")

        # Extract metrics
        fraud_metrics = class_report[fraud_key]

        # Generate the report
        report = {
            'model_performance': {
                'best_hyperparameters': training_result.get('best_params', {}),
                'cross_val_f1': training_result.get('best_score', 0),
                'test_f1': best_f1,
                'precision': fraud_metrics.get('precision', 0),
                'recall': fraud_metrics.get('recall', 0),
                'roc_auc': eval_result.get('roc_curve', {}).get('auc', 0),
                'avg_precision': eval_result.get('precision_recall', {}).get('average_precision', 0)
            },
            'confusion_matrix': optimal_eval.get('confusion_matrix', [[0, 0], [0, 0]]).tolist(),
            'threshold_analysis': {
                'optimal_threshold': best_threshold,
                'false_positives': optimal_eval.get('confusion_matrix', [[0, 0], [0, 0]])[0][1],
                'false_negatives': optimal_eval.get('confusion_matrix', [[0, 0], [0, 0]])[1][0]
            },
            'recommendations': [
                f"Deploy model with threshold set to {best_threshold:.2f}",
                "Monitor false positive rate weekly",
                "Consider feature engineering improvements"
            ],
            'diagnostic_info': {
                'classification_report_keys': list(class_report.keys()),
                'detected_fraud_key': fraud_key
            }
        }

        return report

    except Exception as e:
        error_report = {
            'error': str(e),
            'available_data': {
                'eval_result_keys': list(eval_result.keys()) if isinstance(eval_result, dict) else str(type(eval_result)),
                'training_result_keys': list(training_result.keys()) if isinstance(training_result, dict) else str(type(training_result))
            }
        }
        raise ValueError(f"Failed to generate report: {error_report}") from e