"""
Script to train a Logistic Regression model (L2 penalty) with automated feature engineering.
Outputs probabilities (like Linear Regression) for comparable evaluation.
"""

import sys
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer

# Local imports
from preprocess_data import get_feature_groups, build_pipeline
from evaluate_probabilities import print_report, get_probability_scoring_functions

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
    Main training workflow for Logistic Regression.
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
    
    # Logistic Regression with L2 penalty (Ridge equivalent for classification)
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))
    ])
    
    # 4. Train
    logger.info("Training Logistic Regression model...")
    model_pipeline.fit(X_train, y_train)
    

    # 5. Evaluate
    logger.info("Running 5-Fold Cross-Validation (RMSE, Sensitivity, Specificity, F1, AUC)...")
    
    scoring = get_probability_scoring_functions()
    cv_results = cross_validate(model_pipeline, X_train, y_train, cv=5, scoring=scoring)
    
    logger.info("Predicting on test set (internal split)...")
    # Get probabilities for class 1
    y_pred = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Print Report (reusing the common reporting function)
    # y_pred are probabilities here, which fits the reporting logic
    print_report(target_col, "LogisticRegression (L2)", y_test, y_pred, cv_results)
    
    return model_pipeline

def predict_and_save(model, test_file_path, output_name, id_col):
    """
    Loads test data, predicts PROBABILITIES, and saves.
    """
    logger.info(f"Loading test data from {test_file_path}...")
    df_test = load_data(test_file_path)
    
    logger.info("Generating predictions (probabilities) on test data...")
    # Generate probabilities for class 1
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
    TEST_DATA_PATH = 'data/test.csv'  # Set to None if no test data
    TARGET_COL = 'diagnosed_diabetes'
    ID_COL = 'id'
    OUTPUT_FILENAME = "test_data_logistic_regression_output.csv"
    
    # 1. Train Model
    train_df = load_data(TRAIN_DATA_PATH)
    trained_model = train_and_evaluate(train_df, target_col=TARGET_COL, id_col=ID_COL)
    
    # 2. Predict on Test Data (if provided)
    if TEST_DATA_PATH:
        predict_and_save(trained_model, TEST_DATA_PATH, OUTPUT_FILENAME, id_col=ID_COL)
