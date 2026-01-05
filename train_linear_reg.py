"""
Script to train a Linear Regression model with automated feature engineering.
Utilizes sklearn Pipeline and ColumnTransformer.
Follows local exploration patterns from explore.ipynb.
"""

import sys
import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, recall_score, confusion_matrix

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

def get_feature_groups(df, target_col, id_col='id', unique_threshold=10):
    """
    Identifies column groups for preprocessing.
    """
    # All features excluding target and ID (if present in df)
    # Be robust if target_col is absent (e.g. test data)
    cols_to_exclude = [id_col]
    if target_col in df.columns:
        cols_to_exclude.append(target_col)
        
    features = [c for c in df.columns if c not in cols_to_exclude]
    
    # Identify string/object columns which are strictly categorical
    string_categorical_cols = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Identify numeric columns
    numeric_cols = df[features].select_dtypes(include=['number']).columns.tolist()
    
    numeric_continuous_cols = []
    numeric_ordinal_cols = []
    
    for col in numeric_cols:
        # Check cardinality
        if df[col].nunique() < unique_threshold:
            numeric_ordinal_cols.append(col)
        else:
            numeric_continuous_cols.append(col)
            
    logger.info(f"Continuous Numeric Features ({len(numeric_continuous_cols)}): {numeric_continuous_cols}")
    logger.info(f"Ordinal Numeric Features ( < {unique_threshold} unique) ({len(numeric_ordinal_cols)}): {numeric_ordinal_cols}")
    logger.info(f"Categorical String Features ({len(string_categorical_cols)}): {string_categorical_cols}")
            
    return numeric_continuous_cols, numeric_ordinal_cols, string_categorical_cols

def build_pipeline(num_continuous_cols, num_ordinal_cols, cat_string_cols):
    """
    Builds the sklearn model pipeline.
    """
    # 1. Continuous Numeric -> Standard Scaler
    pipe_continuous = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # 2. Ordinal Numeric -> Ordinal Encoder
    pipe_ordinal = Pipeline([
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # 3. String Categorical -> OneHot Encoder
    pipe_categorical = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', pipe_continuous, num_continuous_cols),
            ('ordinal', pipe_ordinal, num_ordinal_cols),
            ('categorical', pipe_categorical, cat_string_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    
    return pipeline

def train_and_evaluate(df, target_col='diagnosed_diabetes', id_col='id'):
    """
    Main training workflow.
    """
    # 1. Feature Identification
    num_cont, num_ord, cat_str = get_feature_groups(df, target_col, id_col)
    
    # 2. Split Data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 3. Build Model
    model_pipeline = build_pipeline(num_cont, num_ord, cat_str)
    
    # 4. Train
    logger.info("Training Linear Regression model...")
    model_pipeline.fit(X_train, y_train)
    
    # 5. Evaluate
    logger.info("Running 5-Fold Cross-Validation (RMSE, Sensitivity, Specificity)...")
    
    def sensitivity_func(y_true, y_pred):
        y_pred_bin = (y_pred > 0.5).astype(int)
        return recall_score(y_true, y_pred_bin, zero_division=0)

    def specificity_func(y_true, y_pred):
        y_pred_bin = (y_pred > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    scoring = {
        'rmse': 'neg_mean_squared_error',
        'sensitivity': make_scorer(sensitivity_func),
        'specificity': make_scorer(specificity_func)
    }

    cv_results = cross_validate(model_pipeline, X_train, y_train, cv=5, scoring=scoring)
    
    cv_rmse = np.sqrt(-cv_results['test_rmse'])
    cv_sens = cv_results['test_sensitivity']
    cv_spec = cv_results['test_specificity']
    
    logger.info("Predicting on test set (internal split)...")
    y_pred = model_pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("       MODEL PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Target Variable : {target_col}")
    print(f"Model Type      : LinearRegression")
    print(f"RMSE (Holdout)  : {rmse:.4f}")
    print(f"R2 Score        : {r2:.4f}")
    print("-"*40)
    print(f"5-Fold CV RMSE  : {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    print(f"5-Fold CV Sens  : {cv_sens.mean():.4f} (+/- {cv_sens.std() * 2:.4f})")
    print(f"5-Fold CV Spec  : {cv_spec.mean():.4f} (+/- {cv_spec.std() * 2:.4f})")
    print("="*40 + "\n")
    
    return model_pipeline

def predict_and_save(model, test_file_path, output_name, id_col='id'):
    """
    Loads test data, predicts, and saves.
    """
    logger.info(f"Loading test data from {test_file_path}...")
    df_test = load_data(test_file_path)
    
    logger.info("Generating predictions on test data...")
    predictions = model.predict(df_test)
    
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
    OUTPUT_FILENAME = "test_data_linear_regression_output.csv"
    
    # 1. Train Model
    train_df = load_data(TRAIN_DATA_PATH)
    trained_model = train_and_evaluate(train_df, target_col=TARGET_COL, id_col=ID_COL)
    
    # 2. Predict on Test Data (if provided)
    if TEST_DATA_PATH:
        predict_and_save(trained_model, TEST_DATA_PATH, OUTPUT_FILENAME, id_col=ID_COL)
