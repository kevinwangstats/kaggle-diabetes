
import logging
import os
import sys
import pandas as pd
from google.cloud import aiplatform
from google.cloud import storage
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
# GCP Project ID - Assumes this variable is explicitly set or passed in. 
# Defaults to a placeholder if not set by environment variable.
PROJECT_ID = os.getenv("GCP_PROJECT", "project_id") 
LOCATION = "us-central1" # Default region

# Training configuration
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
TARGET_COLUMN = "diagnosed_diabetes"
ID_COLUMN = "id"
TRAIN_BUDGET_MILLI_NODE_HOURS = 100 
MODEL_DISPLAY_NAME = "diabetes_automl_model"
OUTPUT_PREDICTION_FILE = "vertex_automl_predictions.csv"

# -------------------------------------------------------------------------
# AUTHENTICATION NOTE
# -------------------------------------------------------------------------
# This script assumes that the environment is already authenticated.
# 
# Usage:
# 1. Run `gcloud auth login` to authenticate with your user account.
# 2. Run `gcloud auth application-default login` to set up Application Default Credentials (ADC).
# 3. Ensure your user has permissions for Vertex AI (AI Platform Admin) and GCS (Storage Admin).
# -------------------------------------------------------------------------

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return f"gs://{bucket_name}/{destination_blob_name}"

def create_bucket_if_not_exists(bucket_name):
    """Creates a GCS bucket if it doesn't verify existence."""
    storage_client = storage.Client(project=PROJECT_ID)
    try:
        bucket = storage_client.get_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} exists.")
    except Exception:
        logger.info(f"Creating bucket {bucket_name}...")
        bucket = storage_client.create_bucket(bucket_name, location=LOCATION)
    return bucket

def run_vertex_automl_pipeline():
    # 1. Initialize Vertex AI SDK
    logger.info(f"Initializing Vertex AI for project {PROJECT_ID} in {LOCATION}...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # 2. Prepare Data
    # Referencing preprocess_data.py logic:
    # "Ordinal" columns are identified as numeric columns with low cardinality (<10).
    # Since we want to train an AutoML model, identifying these as Numeric (int) preserves their order.
    # If we converted them to strings, Vertex might treat them as categorical (unordered).
    # Therefore, we keep the raw numeric values to respect the "Ordinal" nature.
    # We do NOT perform OneHotEncoder or StandardScaler here; AutoML handles normalization and categorical encoding.
    
    logger.info("Loading training data to prepare for upload...")
    if not os.path.exists(TRAIN_DATA_PATH):
        logger.error(f"Training file not found: {TRAIN_DATA_PATH}")
        sys.exit(1)

    df_train = pd.read_csv(TRAIN_DATA_PATH)
    
    # Drop ID column from training data as it has no predictive power and can confuse the model
    if ID_COLUMN in df_train.columns:
        logger.info(f"Dropping '{ID_COLUMN}' from training data.")
        df_train = df_train.drop(columns=[ID_COLUMN])
    
    # Save a temporary CSV for upload
    temp_train_file = "temp_train_for_vertex.csv"
    df_train.to_csv(temp_train_file, index=False)

    # 3. Upload Data to GCS
    # We need a staging bucket. We'll derive one from the project ID.
    bucket_name = f"{PROJECT_ID}-vertex-staging"
    create_bucket_if_not_exists(bucket_name)
    
    gcs_train_uri = upload_to_gcs(bucket_name, temp_train_file, "data/train.csv")
    logger.info(f"Uploaded training data to {gcs_train_uri}")
    
    # Clean up temp file
    os.remove(temp_train_file)

    # 4. Create Managed Dataset in Vertex AI
    logger.info("Creating Vertex AI TabularDataset...")
    dataset = aiplatform.TabularDataset.create(
        display_name="diabetes_dataset",
        gcs_source=[gcs_train_uri]
    )
    logger.info(f"Dataset created: {dataset.resource_name}")

    # 5. Define and Run AutoML Training Job
    logger.info("Defining AutoML Tabular Training Job...")
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=MODEL_DISPLAY_NAME,
        optimization_prediction_type="classification",
        optimization_objective="minimize-log-loss", # Standard for classification probabilities, minimizes log loss
    )

    logger.info(f"Starting training with budget: {TRAIN_BUDGET_MILLI_NODE_HOURS} milli-node hours (1 hour)")
    model = job.run(
        dataset=dataset,
        target_column=TARGET_COLUMN,
        budget_milli_node_hours=TRAIN_BUDGET_MILLI_NODE_HOURS,
        model_display_name=MODEL_DISPLAY_NAME
    )
    
    logger.info(f"Model training complete. Model resource: {model.resource_name}")

    # 6. Make Predictions on Test Data
    if os.path.exists(TEST_DATA_PATH):
        logger.info("Processing test data for batch prediction...")
        df_test = pd.read_csv(TEST_DATA_PATH)
        
        # Keep IDs for final output
        test_ids = df_test[ID_COLUMN] if ID_COLUMN in df_test.columns else df_test.index
        
        # Prepare test data for upload (drop ID to match schema if necessary, though AutoML usually ignores extra columns locally? 
        # For batch prediction, schema should match training features).
        # We drop ID from input to batch prediction to ensure clean matching.
        # We also usually need to ensure the target column is NOT present (or ignored).
        cols_to_drop = [ID_COLUMN]
        if TARGET_COLUMN in df_test.columns:
            cols_to_drop.append(TARGET_COLUMN)
            
        df_test_input = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
        
        temp_test_file = "temp_test_for_vertex.csv"
        df_test_input.to_csv(temp_test_file, index=False)
        
        gcs_test_uri = upload_to_gcs(bucket_name, temp_test_file, "data/test.csv")
        logger.info(f"Uploaded test data to {gcs_test_uri}")
        os.remove(temp_test_file)

        logger.info("Starting Batch Prediction...")
        
        # Create a folder in GCS for outputs
        output_uri_prefix = f"gs://{bucket_name}/predictions/{uuid.uuid4()}"
        
        # Returns a BatchPredictionJob object
        batch_prediction_job = model.batch_predict(
            job_display_name="diabetes_batch_predict",
            gcs_source=[gcs_test_uri],
            gcs_destination_prefix=output_uri_prefix,
            instances_format='csv',
            predictions_format='csv',
            sync=True 
        )
        
        logger.info(f"Batch prediction job complete: {batch_prediction_job.resource_name}")
        
        # 7. Download and Process Results
        logger.info("Downloading results...")
        
        # The output files will be in the dedicated GCS folder.
        # There might be multiple shards given it's distributed, though small data usually is 1 shard.
        # We need to iterate and fetch.
        storage_client = storage.Client(project=PROJECT_ID)
        # Parse the output directory from the job resource if available, or just use our known prefix
        # batch_prediction_job.output_info.gcs_output_directory gives the exact folder
        try:
             gcs_output_dir = batch_prediction_job.output_info.gcs_output_directory
             prefix = gcs_output_dir.split(f"gs://{bucket_name}/")[1]
        except:
             # Fallback if attribute access fails or structure differs
             prefix = output_uri_prefix.split(f"gs://{bucket_name}/")[1]

        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
        
        prediction_dfs = []
        for blob in blobs:
            if blob.name.endswith("prediction-1.csv") or blob.name.endswith(".csv"): # Pattern matching for result files
                # Download to temp
                local_pred_file = f"temp_pred_{uuid.uuid4()}.csv"
                blob.download_to_filename(local_pred_file)
                prediction_dfs.append(pd.read_csv(local_pred_file))
                os.remove(local_pred_file)
        
        if prediction_dfs:
            # Concatenate all shards
            full_preds = pd.concat(prediction_dfs, ignore_index=True)
            
            # Note regarding row ordering:
            # Vertex AI Batch Prediction with CSV input/output does NOT guarantee 100% preserving row order if sharded.
            # However, for single-shard outputs on small datasets, it often matches.
            # In a strict production environment, we should use 'jsonl' format with 'key' field to join.
            # For this script, we assume a simple join if lengths match.
            
            if len(full_preds) == len(df_test):
                logger.info("Merging predictions with test IDs...")
                
                output_df = pd.DataFrame()
                output_df[ID_COLUMN] = df_test[ID_COLUMN] 
                
                # Append all prediction columns (probabilities/classes)
                final_output = pd.concat([output_df.reset_index(drop=True), full_preds.reset_index(drop=True)], axis=1)
                
                final_output.to_csv(OUTPUT_PREDICTION_FILE, index=False)
                logger.info(f"Predictions saved to local file: {OUTPUT_PREDICTION_FILE}")
            else:
                logger.error(f"Mismatch in prediction row count ({len(full_preds)}) vs input row count ({len(df_test)}).")
        else:
            logger.error("No prediction files found in GCS output directory.")

if __name__ == "__main__":
    run_vertex_automl_pipeline()
