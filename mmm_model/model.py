# model.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU access

import logging
import pandas as pd
import jax.numpy as jnp
import numpyro
import pickle
import datetime
import csv
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightweight_mmm import (
    lightweight_mmm,
    preprocessing
)
from google.cloud import storage
from google.cloud import bigquery
import google.auth

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the training job.
    """
    try:
        # Get default credentials and project ID
        credentials, PROJECT_ID = google.auth.default()
        logger.info(f"Using Project ID: {PROJECT_ID}")

        # BigQuery dataset and table
        BQ_DATASET_TABLE = "data.mmm_weekly_source"

        # Base path for models and metrics in GCS
        MODEL_BASE_GCS_PATH = f"gs://lightweight-mmm-pipeline/models"

        # Number of data points to reserve for testing
        TEST_SIZE = 30

        # Initialize the GCS client
        storage_client = storage.Client()
        logger.info("Initialized Google Cloud Storage client.")

        # Initialize the BigQuery client
        bigquery_client = bigquery.Client(project=PROJECT_ID)
        logger.info("Initialized BigQuery client.")

        # Load and Inspect Data
        df = download_data_from_bigquery(bigquery_client, PROJECT_ID, BQ_DATASET_TABLE)
        logger.info("Data downloaded from BigQuery successfully.")
        logger.info("First few rows of the DataFrame:")
        logger.info("\n%s", df.head())

        # Data Type Checks and Conversions
        media_columns = [
            'google_ads',
            'tiktok_spend',
            'facebook_spend',
            'print_spend',
            'ooh_spend',
            'tv_spend',
            'podcast_radio_spend'
        ]

        numeric_columns = media_columns + ['revenue']

        # Convert columns to numeric types
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values by filling with zeros
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Convert all media columns to float32
        df[media_columns] = df[media_columns].astype(np.float32)

        # Ensure 'revenue' is also of type float32
        df['revenue'] = df['revenue'].astype(np.float32)

        logger.info("\nDataFrame dtypes after conversion:")
        logger.info("\n%s", df.dtypes)

        # Prepare Media Data, Target, and Costs
        media_data = df[media_columns].to_numpy(dtype=np.float32)
        target = df[['revenue']].to_numpy(dtype=np.float32)
        costs = df[media_columns].sum().to_numpy(dtype=np.float32)
        data_size = media_data.shape[0]

        # Split Data into Training and Testing Sets
        split_point = data_size - TEST_SIZE
        media_data_train = media_data[:split_point, ...]
        media_data_test = media_data[split_point:, ...]
        target_train = target[:split_point].reshape(-1)
        target_test = target[split_point:].reshape(-1)

        # Scale the Data
        media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
        target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
        cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

        media_data_train_scaled = media_scaler.fit_transform(media_data_train)
        media_data_test_scaled = media_scaler.transform(media_data_test)
        target_train_scaled = target_scaler.fit_transform(target_train)
        costs_scaled = cost_scaler.fit_transform(costs)

        # Initialize the LightweightMMM Model
        mmm_model = lightweight_mmm.LightweightMMM(model_name="carryover")

        # Define Model Hyperparameters (Adjusted for Cloud Run constraints)
        number_warmup = 500    # Adjusted to ensure timely execution
        number_samples = 500   # Adjusted to ensure timely execution
        number_chains = 1

        # Fit the Model to the Training Data
        logger.info("\nFitting the model... This may take some time.")
        mmm_model.fit(
            media=media_data_train_scaled,
            media_prior=costs_scaled,
            target=target_train_scaled,
            number_warmup=number_warmup,
            number_samples=number_samples,
            number_chains=number_chains,
        )
        logger.info("Model fitting completed.")

        # Evaluate the Model on Test Data
        logger.info("\nMaking predictions on test data...")
        predictions_scaled = mmm_model.predict(media=media_data_test_scaled)
        predictions_scaled_mean = predictions_scaled.mean(axis=0) if predictions_scaled.ndim > 1 else predictions_scaled
        predictions = target_scaler.inverse_transform(predictions_scaled_mean)

        mae = mean_absolute_error(target_test, predictions)
        mse = mean_squared_error(target_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_test, predictions)

        logger.info("\nModel Evaluation on Test Data:")
        logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logger.info(f"R-squared (R²): {r2:.2f}")

        # Save the Trained Model and Scalers to GCS
        version_id = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        MODEL_VERSIONED_FOLDER = f"{MODEL_BASE_GCS_PATH}/version_{version_id}"
        MODEL_LATEST_FOLDER = f"{MODEL_BASE_GCS_PATH}/latest_version"

        local_model_file = "model_and_scalers.pkl"
        model_and_scalers = {
            'model': mmm_model,
            'media_scaler': media_scaler,
            'target_scaler': target_scaler,
            'cost_scaler': cost_scaler
        }

        with open(local_model_file, 'wb') as f:
            pickle.dump(model_and_scalers, f)
        logger.info("Model and scalers saved locally.")

        MODEL_VERSIONED_GCS_PATH = f"{MODEL_VERSIONED_FOLDER}/model_and_scalers.pkl"
        upload_file_to_gcs(storage_client, local_model_file, MODEL_VERSIONED_GCS_PATH)
        logger.info(f"Model uploaded to: {MODEL_VERSIONED_GCS_PATH}")

        MODEL_LATEST_GCS_PATH = f"{MODEL_LATEST_FOLDER}/model_and_scalers.pkl"
        upload_file_to_gcs(storage_client, local_model_file, MODEL_LATEST_GCS_PATH)
        logger.info(f"Model uploaded to: {MODEL_LATEST_GCS_PATH}")

        # Save Evaluation Metrics to GCS as CSV
        metrics_dict = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'model_version': version_id
        }

        local_metrics_file = "metrics.csv"
        save_metrics_to_csv(metrics_dict, local_metrics_file)
        logger.info("Metrics saved locally.")

        METRICS_VERSIONED_GCS_PATH = f"{MODEL_VERSIONED_FOLDER}/metrics.csv"
        upload_file_to_gcs(storage_client, local_metrics_file, METRICS_VERSIONED_GCS_PATH)
        logger.info(f"Metrics uploaded to: {METRICS_VERSIONED_GCS_PATH}")

        METRICS_LATEST_GCS_PATH = f"{MODEL_LATEST_FOLDER}/metrics.csv"
        upload_file_to_gcs(storage_client, local_metrics_file, METRICS_LATEST_GCS_PATH)
        logger.info(f"Metrics uploaded to: {METRICS_LATEST_GCS_PATH}")

        # Clean up local files
        os.remove(local_model_file)
        os.remove(local_metrics_file)
        logger.info("Local files have been cleaned up.")

        logger.info("Training job completed successfully.")

    except Exception as e:
        logger.error("An error occurred during the training job.", exc_info=True)
        raise e  # Re-raise the exception to ensure the job reports failure

def download_data_from_bigquery(client, project_id, dataset_table):
    """
    Downloads data from BigQuery and returns it as a pandas DataFrame.
    """
    query = f"""
    SELECT * FROM `{project_id}.{dataset_table}`
    """
    df = client.query(query).to_dataframe()
    return df

def upload_file_to_gcs(storage_client, local_file_path, gcs_path):
    """
    Uploads a local file to GCS.
    """
    bucket_name, blob_path = parse_gcs_path(gcs_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Upload the local file to GCS
    blob.upload_from_filename(local_file_path)
    logger.info(f"Uploaded {local_file_path} to {gcs_path}")

def parse_gcs_path(gcs_path):
    """
    Parses a GCS path into bucket name and blob path.
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError("GCS path must start with 'gs://'")
    parts = gcs_path[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError("GCS path must be in the format 'gs://bucket_name/path/to/blob'")
    return parts[0], parts[1]

def save_metrics_to_csv(metrics_dict, local_file_path):
    """
    Saves the metrics dictionary to a CSV file.
    """
    # Check if file exists
    file_exists = os.path.isfile(local_file_path)

    # Define CSV headers
    headers = ['timestamp'] + list(metrics_dict.keys())

    # Add timestamp to metrics
    metrics_with_timestamp = {'timestamp': datetime.datetime.utcnow().isoformat(), **metrics_dict}

    # Write metrics to CSV
    with open(local_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_with_timestamp)
    logger.info(f"Metrics saved to {local_file_path}")

if __name__ == '__main__':
    main()
