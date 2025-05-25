# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from imblearn.over_sampling import SMOTE
# from config.config import RANDOM_STATE, TEST_SIZE
#
# def train_model(X, y):
#     """Train Random Forest model with hyperparameter tuning"""
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
#     )
#
#     # Handle class imbalance
#     smote = SMOTE(random_state=RANDOM_STATE)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#
#     # Model training with GridSearchCV
#     rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
#
#     param_grid = {
#         'n_estimators': [50, 100, 200], # Number of trees (balance speed/accuracy)
#         'max_depth': [None, 10, 20, 30], # Trees grow until pure
#         'min_samples_split': [2, 5, 10], # Minimum samples to split a node
#         'min_samples_leaf': [1, 2, 4] # Controls overfitting
#     }
#
#     grid_search = GridSearchCV(
#         rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1
#     )
#     grid_search.fit(X_train_resampled, y_train_resampled)
#
#     return {
#         'model': grid_search.best_estimator_,
#         'X_test': X_test,
#         'y_test': y_test,
#         'best_params': grid_search.best_params_,
#         'best_score': grid_search.best_score_
#     }

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE

# Constants
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_EVALS = 50  # Number of optimization iterations

def train_model(X, y, model_type='xgb'):
    """Train XGBoost/LightGBM with Bayesian Optimization (Hyperopt)"""

    # Split data (stratified due to imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # SMOTE for class imbalance
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define objective function for Hyperopt
    def objective(params):
        if model_type == 'xgb':
            model = XGBClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=RANDOM_STATE,
                scale_pos_weight=len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])
            )
        elif model_type == 'lgbm':
            model = LGBMClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=RANDOM_STATE,
                is_unbalance=True  # Handles imbalance internally
            )

        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        return {'loss': -f1, 'status': STATUS_OK}  # Minimize -F1 = Maximize F1

    # Define search space for Hyperopt
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 25),
        'max_depth': hp.quniform('max_depth', 3, 20, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
    }

    # Run Bayesian Optimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials
    )

    # Train final model with best params
    if model_type == 'xgb':
        best_model = XGBClassifier(
            n_estimators=int(best['n_estimators']),
            max_depth=int(best['max_depth']),
            learning_rate=best['learning_rate'],
            subsample=best['subsample'],
            colsample_bytree=best['colsample_bytree'],
            random_state=RANDOM_STATE,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        )
    elif model_type == 'lgbm':
        best_model = LGBMClassifier(
            n_estimators=int(best['n_estimators']),
            max_depth=int(best['max_depth']),
            learning_rate=best['learning_rate'],
            subsample=best['subsample'],
            colsample_bytree=best['colsample_bytree'],
            random_state=RANDOM_STATE,
            is_unbalance=True
        )

    best_model.fit(X_train_resampled, y_train_resampled)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': best_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'best_params': best,
        'f1_score': f1_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }