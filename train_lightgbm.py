"""
Script to train a LightGBM Classifier with hyperparameter tuning using RandomizedSearchCV.
Outputs probabilities for evaluation.
"""

import sys
import logging
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer

# Local imports
from preprocess_data import get_feature_groups, build_pipeline
from evaluate_probabilities import print_report, sensitivity_score, specificity_score

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Loads data from CSV.
    """
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        sys.exit(1)

def train_and_evaluate(df, target_col, id_col):
    """
    Main training workflow for LightGBM with Hyperparameter Tuning.
    """
    # 1. Feature Identification
    num_cont, num_ord, cat_str = get_feature_groups(df, target_col, id_col)
    
    # 2. Split Data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 3. Build Model Pipeline
    preprocessor = build_pipeline(num_cont, num_ord, cat_str)
    
    # LightGBM Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', lgb.LGBMClassifier(random_state=42, verbose=-1))
    ])
    
    # 4. Hyperparameter Tuning
    logger.info("Starting Hyperparameter Tuning with RandomizedSearchCV (5-Fold CV)...")
    
    param_dist = {
        'model__n_estimators': [100, 200, 500],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__num_leaves': [31, 63, 127],
        'model__max_depth': [-1, 10, 20],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'model__reg_alpha': [0, 0.1, 1, 5],
        'model__reg_lambda': [0, 0.1, 1, 5]
    }
    
    # Define scorers
    def proba_rmse(y_true, y_pred_proba):
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        return -mean_squared_error(y_true, y_pred_proba) # Return neg MSE for sqrt later

    def proba_sensitivity(y_true, y_pred_proba):
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        return sensitivity_score(y_true, y_pred_proba)

    def proba_specificity(y_true, y_pred_proba):
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        return specificity_score(y_true, y_pred_proba)

    scoring = {
        'rmse': make_scorer(proba_rmse, response_method='predict_proba'),
        'sensitivity': make_scorer(proba_sensitivity, response_method='predict_proba'),
        'specificity': make_scorer(proba_specificity, response_method='predict_proba')
    }
    
    # Tuning - optimizing for RMSE (neg_mean_squared_error) as it's the competition metric usually?
    # Or just 'neg_log_loss'? The prompt imply "accuracy" or general performance.
    # Given we are evaluating probabilities and RMSE, let's optimize for 'rmse' (neg_mean_squared_error on probs).
    
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=20, # 20 iterations
        cv=5, 
        scoring=scoring,
        refit='rmse', # Refit on the best RMSE score
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_pipeline = random_search.best_estimator_
    logger.info(f"Best Hyperparameters: {random_search.best_params_}")
    logger.info(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")

    # Extract CV results for the best estimator for reporting
    best_index = random_search.best_index_
    cv_results = {
        'test_rmse': np.array([random_search.cv_results_['mean_test_rmse'][best_index]]), # This is already neg MSE
        'test_sensitivity': np.array([random_search.cv_results_['mean_test_sensitivity'][best_index]]),
        'test_specificity': np.array([random_search.cv_results_['mean_test_specificity'][best_index]])
    }
    
    # 5. Evaluate on Test Set
    logger.info("Predicting on test set (internal split)...")
    y_pred = best_pipeline.predict_proba(X_test)[:, 1]
    
    print_report("diagnosed_diabetes", "LightGBM (Tuned)", y_test, y_pred, cv_results)
    
    return best_pipeline

def predict_and_save(model, test_file_path, output_name, id_col):
    """
    Loads test data, predicts PROBABILITIES, and saves.
    """
    logger.info(f"Loading test data from {test_file_path}...")
    df_test = load_data(test_file_path)
    
    logger.info("Generating predictions (probabilities) on test data...")
    predictions = model.predict_proba(df_test)[:, 1]
    
    # Save output
    output_df = pd.DataFrame({
        id_col: df_test[id_col],
        'prediction': predictions
    })
    
    output_dir = os.path.dirname(test_file_path)
    output_path = os.path.join(output_dir, output_name)
    
    logger.info(f"Saving predictions to {output_path}...")
    output_df.to_csv(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    # Hardcoded Configuration
    TRAIN_DATA_PATH = 'data/train.csv'
    TEST_DATA_PATH = 'data/test.csv'
    TARGET_COL = 'diagnosed_diabetes'
    ID_COL = 'id'
    OUTPUT_FILENAME = "test_data_lightgbm_output.csv"
    
    # 1. Train Model
    train_df = load_data(TRAIN_DATA_PATH)
    trained_model = train_and_evaluate(train_df, target_col=TARGET_COL, id_col=ID_COL)
    
    # 2. Predict on Test Data (if provided)
    if TEST_DATA_PATH:
        predict_and_save(trained_model, TEST_DATA_PATH, OUTPUT_FILENAME, id_col=ID_COL)
