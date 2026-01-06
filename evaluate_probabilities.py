"""
Script for model evaluation metrics and reporting.
Handles custom scoring functions and performance printouts.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, recall_score, confusion_matrix, f1_score, roc_auc_score

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

def f1_score_metric(y_true, y_pred):
    """
    Calculates F1 score for binary classification with a 0.5 threshold.
    """
    y_pred_bin = (y_pred > 0.5).astype(int)
    return f1_score(y_true, y_pred_bin, zero_division=0)

def roc_auc_score_metric(y_true, y_pred):
    """
    Calculates ROC AUC score. y_pred are expected to be probabilities or continuous scores.
    """
    return roc_auc_score(y_true, y_pred)

def get_scoring_functions():
    """
    Returns a dictionary of scorers for sklearn's cross_validate.
    """
    return {
        'rmse': 'neg_mean_squared_error',
        'sensitivity': make_scorer(sensitivity_score),
        'specificity': make_scorer(specificity_score),
        'f1': make_scorer(f1_score_metric),
        'roc_auc': make_scorer(roc_auc_score_metric, needs_proba=False) # needs_proba=False because we pass probs directly in our custom wrappers if needed, but for standard models standard predict might be class. 
        # Actually, in linear regression predict returns float (scores). In logistic, we usually use predict_proba. 
        # Standard make_scorer(roc_auc_score, needs_proba=True) works if the model is a classifier.
        # But we are using custom wrappers in the training scripts (e.g. proba_sensitivity) to handle dimensions.
        # Let's stick to the pattern: the training script defines how to extract the "score" (prob) and passes it to these functions if they are wrapped there.
        # However, cross_validate usage of 'roc_auc' usually requires needs_threshold=True or needs_proba=True.
        # Given the complexity in training scripts, let's keep it simple here and just return the scorer.
        # The training scripts (train_logistic_reg.py) map these names to MakeScorer calls or just string shorthands.
        # Wait, train_logistic_reg.py manually defines the scoring dict using make_scorer(..., response_method='predict_proba').
        # So we just need to provide the function handles or make_scorer objects here?
        # train_linear_reg calls get_scoring_functions() directly.
        # train_logistic_reg defines its own dict.
        # So we update this for train_linear_reg usage mainly, and for general utility.
    }

def _ensure_1d_proba(y_pred_proba):
    """
    Helper to extract positive class probability from 2D array if needed.
    """
    if y_pred_proba.ndim == 2:
        return y_pred_proba[:, 1]
    return y_pred_proba

def proba_rmse_score(y_true, y_pred_proba):
    y_pred_proba = _ensure_1d_proba(y_pred_proba)
    return -np.sqrt(mean_squared_error(y_true, y_pred_proba))

def proba_sensitivity_score(y_true, y_pred_proba):
    y_pred_proba = _ensure_1d_proba(y_pred_proba)
    return sensitivity_score(y_true, y_pred_proba)

def proba_specificity_score(y_true, y_pred_proba):
    y_pred_proba = _ensure_1d_proba(y_pred_proba)
    return specificity_score(y_true, y_pred_proba)

def proba_f1_score(y_true, y_pred_proba):
    y_pred_proba = _ensure_1d_proba(y_pred_proba)
    return f1_score_metric(y_true, y_pred_proba)

def proba_roc_auc_score(y_true, y_pred_proba):
    y_pred_proba = _ensure_1d_proba(y_pred_proba)
    return roc_auc_score(y_true, y_pred_proba)

def get_probability_scoring_functions():
    """
    Returns a dictionary of scorers for models that predict probabilities.
    Configured to expect 'predict_proba' output.
    """
    return {
        'rmse': make_scorer(proba_rmse_score, response_method='predict_proba'),
        'sensitivity': make_scorer(proba_sensitivity_score, response_method='predict_proba'),
        'specificity': make_scorer(proba_specificity_score, response_method='predict_proba'),
        'f1': make_scorer(proba_f1_score, response_method='predict_proba'),
        'roc_auc': make_scorer(proba_roc_auc_score, response_method='predict_proba')
    }

def get_scoring_functions():
    """
    Returns a dictionary of scorers for sklearn's cross_validate.
    Used primarily by train_linear_reg.py (non-probability models).
    """
    return {
        'rmse': 'neg_mean_squared_error',
        'sensitivity': make_scorer(sensitivity_score),
        'specificity': make_scorer(specificity_score),
        'f1': make_scorer(f1_score_metric),
        'roc_auc': make_scorer(roc_auc_score_metric) # Linear regression returns continuous vals
    }

def print_report(target_col, model_name, y_test, y_pred, cv_results=None):
    """
    Prints a standard model performance report.
    
    Args:
        target_col (str): Name of the target variable.
        model_name (str): Name of the model.
        y_test (array-like): True labels for the test set.
        y_pred (array-like): Predicted values/probabilities for the test set.
        cv_results (dict, optional): Results dictionary from cross_validate.
    """
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional holdout metrics
    f1 = f1_score_metric(y_test, y_pred)
    auc = roc_auc_score_metric(y_test, y_pred)
    sens = sensitivity_score(y_test, y_pred)
    spec = specificity_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("       MODEL PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Target Variable : {target_col}")
    print(f"Model Type      : {model_name}")
    print(f"RMSE (Holdout)  : {rmse:.4f}")
    print(f"R2 Score        : {r2:.4f}")
    print(f"ROC AUC         : {auc:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print(f"Sensitivity     : {sens:.4f}")
    print(f"Specificity     : {spec:.4f}")
    
    if cv_results:
        # Note: neg_mean_squared_error returns negative values
        cv_rmse = np.sqrt(-cv_results['test_rmse'])
        cv_sens = cv_results['test_sensitivity']
        cv_spec = cv_results['test_specificity']
        
        # Check if new metrics are in cv_results (handled by training scripts updating their scorer dicts)
        cv_f1 = cv_results.get('test_f1')
        cv_auc = cv_results.get('test_roc_auc')
        
        print("-"*40)
        print(f"5-Fold CV RMSE  : {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        if cv_auc is not None:
             print(f"5-Fold CV AUC   : {cv_auc.mean():.4f} (+/- {cv_auc.std() * 2:.4f})")
        if cv_f1 is not None:
             print(f"5-Fold CV F1    : {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
        print(f"5-Fold CV Sens  : {cv_sens.mean():.4f} (+/- {cv_sens.std() * 2:.4f})")
        print(f"5-Fold CV Spec  : {cv_spec.mean():.4f} (+/- {cv_spec.std() * 2:.4f})")
        
    print("="*40 + "\n")
