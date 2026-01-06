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
from sklearn.model_selection import train_test_split

# Local imports
from evaluate_probabilities import print_report

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_h2o():
    """
    Initialize H2O cluster.
    """
    logger.info("Initializing H2O Cluster...")
    # nthreads=-1 autodetects all cores.
    h2o.init(nthreads=-1)

def load_data_as_h2o_frame(filepath):
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

def train_and_compare_models(train_path, target_col='diagnosed_diabetes', id_col='id'):
    """
    Main training workflow using H2O AutoML to compare models.
    """
    # 1. Initialize H2O
    init_h2o()
    
    # 2. Load Data
    full_hf = load_data_as_h2o_frame(train_path)
    
    # 3. Preprocessing
    # Ensure target is a factor for classification
    logger.info(f"Converting target '{target_col}' to categorical (factor)...")
    full_hf[target_col] = full_hf[target_col].asfactor()
    
    # Identify predictors
    predictors = [n for n in full_hf.names if n != target_col and n != id_col]
    logger.info(f"Predictors ({len(predictors)}): {predictors}")
    
    # 4. Split Data (Train/Test) - replicating the 80/20 split methodology
    logger.info("Splitting data into Train (80%) and Test (Holdout) (20%)...")
    # H2O split_frame returns a list of frames
    train_hf, test_hf = full_hf.split_frame(ratios=[0.8], seed=42)
    
    logger.info(f"Train size: {train_hf.shape[0]}, Holdout Test size: {test_hf.shape[0]}")
    
    # 5. Run AutoML
    # AutoML will train GLMs, RFs, GBMs, DeepLearning, and Stacked Ensembles
    logger.info("Running H2O AutoML to compare multiple models...")
    # run for a max number of models or time. 
    # Let's set max_models=20 to give a good variety similar to RandomizedGridSearch
    aml = H2OAutoML(
        max_models=20, 
        seed=42, 
        project_name="diabetes_automl_comparison",
        nfolds=5, # 5-fold CV within the training set
        sort_metric="RMSE", # Optimize for RMSE to match other scripts
        balance_classes=True # Good practice for unbalanced data, though not strictly requested
    )
    
    aml.train(x=predictors, y=target_col, training_frame=train_hf)
    
    # 6. Leaderboard
    logger.info("AutoML Training Completed. Leaderboard:")
    lb = aml.leaderboard
    # Convert to pandas for display
    lb_df = lb.as_data_frame()
    print("\n" + lb_df.head(10).to_string() + "\n")
    
    # 7. Evaluate Best Model on Holdout
    best_model = aml.leader
    logger.info(f"Evaluating best model: {best_model.model_id} on holdout set...")
    
    # Predict returns frame with 'predict' (class), 'p0', 'p1'...
    preds = best_model.predict(test_hf)
    
    # Convert to numpy/pandas for compatibility with report functions
    # 'p1' is the probability of the positive class
    y_pred_proba = preds['p1'].as_data_frame().values.flatten()
    y_test_true = test_hf[target_col].as_data_frame().values.flatten().astype(float) # convert factor back to float for metrics
    
    # Prepare CV results from AutoML leaderboard/model object if possible, 
    # or just report Holdout metrics. 
    # AutoML object calculates CV metrics for the leader.
    cv_rmse = best_model.rmse(xval=True)
    # H2O metrics are complex objects, let's just use the scalar values available nicely
    
    # Create a mock cv_results dict for formatting
    # Note: H2O gives one aggregated CV score, not per-fold array easily accessible without more calls
    cv_results_mock = {
        'test_rmse': np.array([- (cv_rmse**2)]), # Negating because report expects negative MSE
        # Sensitivity/Specificity tough to get directly as scalar without digging into confusion matrices per fold
        'test_sensitivity': np.array([0.0]), # Placeholder
        'test_specificity': np.array([0.0])  # Placeholder
    }
    
    print_report(target_col, f"H2O AutoML - {best_model.model_id}", y_test_true, y_pred_proba, cv_results=None)
    
    return aml, best_model

def predict_and_save(model, test_path, output_name, id_col='id'):
    """
    Loads test data, predicts with H2O model, and saves.
    """
    logger.info(f"Loading test data from {test_path}...")
    test_hf = load_data_as_h2o_frame(test_path)
    
    logger.info(f"Generating predictions using model: {model.model_id}...")
    preds = model.predict(test_hf)
    
    # Extract probas
    predictions = preds['p1'].as_data_frame().iloc[:, 0]
    ids = test_hf[id_col].as_data_frame().iloc[:, 0]
    
    # Save output
    output_df = pd.DataFrame({
        id_col: ids,
        'prediction': predictions
    })
    
    output_dir = os.path.dirname(test_path)
    output_path = os.path.join(output_dir, output_name)
    
    logger.info(f"Saving predictions to {output_path}...")
    output_df.to_csv(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    # Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data/train.csv')
    TEST_DATA_PATH = os.path.join(BASE_DIR, 'data/test.csv')
    OUTPUT_FILENAME = "test_data_h2o_automl_output.csv"
    TARGET_COL = 'diagnosed_diabetes'
    ID_COL = 'id'
    
    # 1. Train and Compare
    aml, best_model = train_and_compare_models(TRAIN_DATA_PATH, target_col=TARGET_COL, id_col=ID_COL)
    
    # 2. Predict
    if os.path.exists(TEST_DATA_PATH):
        predict_and_save(best_model, TEST_DATA_PATH, OUTPUT_FILENAME, id_col=ID_COL)
    else:
        logger.warning(f"Test data not found at {TEST_DATA_PATH}")
