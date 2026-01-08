"""
Script to train and compare multiple machine learning models using H2O AutoML.
Matches the structure of train_linear_reg.py but utilizes H2O's efficient distributed computing.
"""

import sys
import logging
import os
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np

# Local imports
from evaluate_probabilities import print_report

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_h2o():
    """
    Initialize H2O cluster.
    """
    logger.info("Initializing H2O Cluster...")
    # nthreads=-1 autodetects all cores.
    h2o.init(nthreads=-1)

def load_data(filepath):
    """
    Loads data directly into an H2O Frame.
    """
    logger.info(f"Loading data from {filepath} into H2O Frame...")
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        sys.exit(1)
        
    try:
        # h2o.import_file is generally faster and handles types well
        hf = h2o.import_file(path=filepath)
        logger.info(f"Loaded H2O Frame with shape: {hf.shape}")
        return hf
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

def preprocess_data(frame, target_col, id_col):
    """
    Preprocesses data for H2O: converts target to factor, identifies predictors.
    
    Args:
        frame: H2OFrame
        target_col: str
        id_col: str
        
    Returns:
        predictors: list of strings
    """
    # Ensure target is a factor for classification
    logger.info(f"Converting target '{target_col}' to categorical (factor)...")
    frame[target_col] = frame[target_col].asfactor()
    
    # Identify predictors
    predictors = [n for n in frame.names if n != target_col and n != id_col]
    logger.info(f"Predictors ({len(predictors)}): {predictors}")
    
    return predictors

def print_top_models_per_family(aml, top_n=2):
    """
    Prints the top N models for each model family from the leaderboard.
    """
    logger.info(f"Extracting top {top_n} models per family...")
    
    # Get leaderboard as pandas dataframe
    lb = aml.leaderboard.as_data_frame()
    
    # Define common families. Note: 'DRF' includes Random Forest and XRT.
    families = [
        'GLM', 
        'DRF', 
        'GBM', 
        # 'DeepLearning', 
        'StackedEnsemble', 
        'XGBoost',
        ]

    print("\n" + "="*50)
    print(f"Top {top_n} Models per Family")
    print("="*50)
    
    for family in families:
        # Filter for models belonging to the family
        # H2O model IDs usually start with the family name
        family_models = lb[lb['model_id'].str.contains(family, case=False, na=False)]
        
        if not family_models.empty:
            print(f"\nFamily: {family}")
            top_models = family_models.head(top_n)
            
            # Print details
            for idx, row in top_models.iterrows():
                print(f"  Rank {idx + 1} ({family}): {row['model_id']}")
                # Assuming RMSE is the sort metric (first column after model_id usually, or check aml.sort_metric)
                # But safer to print the columns available
                metrics_str = ", ".join([f"{col}: {row[col]:.4f}" for col in lb.columns if col != 'model_id' and isinstance(row[col], (int, float))][:2])
                print(f"      Metrics: {metrics_str}")
        else:
            # Some families might not be present (e.g. XGBoost on Mac sometimes, or if excluded)
            pass

def train_automl(train_frame, target_col, predictors, max_models=20, seed=42, project_name="diabetes_automl"):
    """
    Runs H2O AutoML on the training frame.
    
    Args:
        train_frame: H2OFrame for training
        target_col: str
        predictors: list of str
        max_models: int
        seed: int
        project_name: str
        
    Returns:
        aml: H2OAutoML object
    """
    logger.info(f"Running H2O AutoML with max_models={max_models}...")
    
    aml = H2OAutoML(
        max_models=max_models, 
        seed=seed, 
        project_name=project_name,
        nfolds=5, 
        sort_metric="RMSE", 
        balance_classes=True,
        preprocessing=["target_encoding"] # Enable target encoding for high cardinality categoricals
    )
    
    aml.train(x=predictors, y=target_col, training_frame=train_frame)
    return aml

def evaluate_model(model, test_frame, target_col):
    """
    Evaluates the model on a holdout test frame and prints report.
    """
    logger.info(f"Evaluating model: {model.model_id} on holdout set...")
    
    # Predict returns frame with 'predict' (class), 'p0', 'p1'...
    preds = model.predict(test_frame)
    
    # Convert to numpy/pandas for compatibility with report functions
    # 'p1' is the probability of the positive class
    y_pred_proba = preds['p1'].as_data_frame().values.flatten()
    y_test_true = test_frame[target_col].as_data_frame().values.flatten().astype(float)
    
    # We pass None for cv_results as capturing them from H2O requires extra logic not requested
    print_report(target_col, f"H2O AutoML - {model.model_id}", y_test_true, y_pred_proba, cv_results=None)

def generate_predictions(model, test_frame, id_col):
    """
    Generates predictions for a dataframe.
    """
    logger.info(f"Generating predictions using model: {model.model_id}...")
    preds = model.predict(test_frame)
    
    # Extract probas
    predictions = preds['p1'].as_data_frame().iloc[:, 0]
    ids = test_frame[id_col].as_data_frame().iloc[:, 0]
    
    output_df = pd.DataFrame({
        id_col: ids,
        'prediction': predictions
    })
    return output_df

if __name__ == "__main__":
    # Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data/train.csv')
    TEST_DATA_PATH = os.path.join(BASE_DIR, 'data/test.csv')
    OUTPUT_FILENAME = "test_data_h2o_automl_output.csv"
    TARGET_COL = 'diagnosed_diabetes'
    ID_COL = 'id'
    MAX_MODELS = 20  # Set the maximum number of models in total to compute
    
    # 1. Initialize
    start_h2o()
    
    # 2. Load Data
    full_hf = load_data(TRAIN_DATA_PATH)
    
    # 3. Preprocess
    predictors = preprocess_data(full_hf, TARGET_COL, ID_COL)
    
    # 4. Split Data
    logger.info("Splitting data into Train (80%) and Test (Holdout) (20%)...")
    train_hf, test_hf = full_hf.split_frame(ratios=[0.8], seed=42)
    logger.info(f"Train size: {train_hf.shape[0]}, Holdout Test size: {test_hf.shape[0]}")
    
    # 5. Train AutoML
    # Use max_models to control the total number of models computed
    aml = train_automl(train_hf, TARGET_COL, predictors, max_models=MAX_MODELS)
    
    # 6. Leaderboard
    logger.info("AutoML Training Completed. Leaderboard:")
    lb = aml.leaderboard
    lb_df = lb.as_data_frame()
    print("\n" + lb_df.head(10).to_string() + "\n")
    
    # 6.1 Print Top Models per Family
    print_top_models_per_family(aml)
    
    # 7. Evaluate
    best_model = aml.leader
    
    print("\n" + "="*50)
    print(f"Best Model Found: {best_model.model_id}")
    print("="*50)
    best_model.show()
    
    evaluate_model(best_model, test_hf, TARGET_COL)
    
    # 8. Predict on Test Data (if exists)
    if os.path.exists(TEST_DATA_PATH):
        logger.info(f"Loading test data from {TEST_DATA_PATH}...")
        test_data_hf = load_data(TEST_DATA_PATH)
        output_df = generate_predictions(best_model, test_data_hf, ID_COL)
        
        output_dir = os.path.dirname(TEST_DATA_PATH)
        output_path = os.path.join(output_dir, OUTPUT_FILENAME)
        
        logger.info(f"Saving predictions to {output_path}...")
        output_df.to_csv(output_path, index=False)
        logger.info("Done.")
    else:
        logger.warning(f"Test data not found at {TEST_DATA_PATH}")
