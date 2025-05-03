import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score
)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model performance"""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Generate evaluation metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'precision_recall': {
            'precision': precision,
            'recall': recall,
            'average_precision': avg_precision
        },
        'y_prob': y_prob,
        'y_pred': y_pred
    }

def find_optimal_threshold(y_test, y_prob):
    """Find optimal threshold based on F1 score"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]
