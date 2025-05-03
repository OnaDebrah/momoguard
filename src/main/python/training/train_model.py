from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from config.config import RANDOM_STATE, TEST_SIZE

def train_model(X, y):
    """Train Random Forest model with hyperparameter tuning"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Handle class imbalance
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Model training with GridSearchCV
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)

    param_grid = {
        'n_estimators': [50, 100, 200], # Number of trees (balance speed/accuracy)
        'max_depth': [None, 10, 20, 30], # Trees grow until pure
        'min_samples_split': [2, 5, 10], # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 4] # Controls overfitting
    }

    grid_search = GridSearchCV(
        rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)

    return {
        'model': grid_search.best_estimator_,
        'X_test': X_test,
        'y_test': y_test,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }