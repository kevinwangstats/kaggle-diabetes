"""
Script for model evaluation metrics and reporting.
Handles custom scoring functions and performance printouts.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, recall_score, confusion_matrix

def sensitivity_score(y_true, y_pred):
    """
    Calculates sensitivity (recall) for binary classification with a 0.5 threshold.
    For Linear Regression, y_pred are continuous values.
    """
    y_pred_bin = (y_pred > 0.5).astype(int)
    return recall_score(y_true, y_pred_bin, zero_division=0)

def specificity_score(y_true, y_pred):
    """
    Calculates specificity (True Negative Rate) for binary classification with a 0.5 threshold.
    """
    y_pred_bin = (y_pred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def get_scoring_functions():
    """
    Returns a dictionary of scorers for sklearn's cross_validate.
    """
    return {
        'rmse': 'neg_mean_squared_error',
        'sensitivity': make_scorer(sensitivity_score),
        'specificity': make_scorer(specificity_score)
    }

def print_report(target_col, model_name, y_test, y_pred, cv_results=None):
    """
    Prints a standard model performance report.
    
    Args:
        target_col (str): Name of the target variable.
        model_name (str): Name of the model.
        y_test (array-like): True labels for the test set.
        y_pred (array-like): Predicted values for the test set.
        cv_results (dict, optional): Results dictionary from cross_validate.
    """
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("       MODEL PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Target Variable : {target_col}")
    print(f"Model Type      : {model_name}")
    print(f"RMSE (Holdout)  : {rmse:.4f}")
    print(f"R2 Score        : {r2:.4f}")
    
    if cv_results:
        # Note: neg_mean_squared_error returns negative values
        cv_rmse = np.sqrt(-cv_results['test_rmse'])
        cv_sens = cv_results['test_sensitivity']
        cv_spec = cv_results['test_specificity']
        
        print("-"*40)
        print(f"5-Fold CV RMSE  : {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        print(f"5-Fold CV Sens  : {cv_sens.mean():.4f} (+/- {cv_sens.std() * 2:.4f})")
        print(f"5-Fold CV Spec  : {cv_spec.mean():.4f} (+/- {cv_spec.std() * 2:.4f})")
        
    print("="*40 + "\n")
