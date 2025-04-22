"""
This run can be run directly using:
        uv run evaluate.py

Arguments can be passed using the command line, see --help and parse_args() for more details.

"""

import argparse
import logging
import os
import traceback

import httpx
import mlflow
import pandas as pd
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from constants import API_DEPLOYED_URL
from utils.data import get_df_naf, get_file_system

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


def load_test_data(num_samples):
    fs = get_file_system()
    df_test = pd.read_parquet(
        "projet-ape/model_comparison_splits/sirene_4_NAF2025_20241027/df_test_raw.parquet",
        filesystem=fs,
    )[["nace2025", "libelle"]]

    return df_test.iloc[:num_samples].reset_index(drop=True)


def get_all_levels(df, df_naf, col):
    """
    Get all levels of the NAF code from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the NAF codes.
        df_naf (pd.DataFrame): DataFrame containing the NAF data.

    Returns:
        pd.DataFrame: DataFrame with all levels of the NAF code.
    """
    df = df[[col]]
    df_full = df.merge(df_naf, left_on=col, right_on="APE_NIV5").drop(columns=[col, "LIB_NIV5"])

    return df_full


def process_response(response):
    """
    Get json response from the API and return predictions and probabilites Numpy arrays.
    """
    preds = pd.DataFrame(response)
    preds["code_ape"] = preds["code_ape"].str.replace(".", "", regex=False)

    return preds


def save_response(df_res, entry_point: str):
    fs = get_file_system()
    file_path = f"projet-ape/model_comparison_splits/sirene_4_NAF2025_20241027/df_test_raw_pred_{entry_point}.parquet"
    with fs.open(file_path, "wb") as f:
        df_res.to_parquet(f)

    return df_res


def create_or_restore_experiment(experiment_name):
    client = MlflowClient()

    try:
        # Check if the experiment exists (either active or deleted)
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment:
            if experiment.lifecycle_stage == "deleted":
                # Restore the experiment if it's deleted
                print(f"Restoring deleted experiment: '{experiment_name}' (ID: {experiment.experiment_id})")
                client.restore_experiment(experiment.experiment_id)
            else:
                print(f"Experiment '{experiment_name}' already exists and is active (ID: {experiment.experiment_id}).")
        else:
            # Create the experiment if it doesn't exist
            print(f"Creating a new experiment: '{experiment_name}'")
            experiment_id = client.create_experiment(experiment_name)
            print(f"Created experiment '{experiment_name}' with ID: {experiment_id}")

    except RestException as e:
        print(f"An error occurred while handling the experiment '{experiment_name}': {e}")


def evaluate(df_naf, exp_name: str, api_inputs: dict, entry_point: str, num_samples=1000):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    create_or_restore_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run():
        mlflow.log_param("entry_point", entry_point)
        response = httpx.post(f"{API_DEPLOYED_URL}/{entry_point}/batch", json=api_inputs, timeout=600)
        response.raise_for_status()
        results = response.json()

        preds = process_response(results)
        preds = get_all_levels(preds, df_naf, col="code_ape")

        # Accuracies at all levels
        accs = (preds == ground_truth).mean(axis=0)

        for i in range(len(accs)):
            mlflow.log_metric(f"accuracy_lvl_{i + 1}", accs[i])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument(
        "--entry_point",
        type=str,
        default="all",
        choices=["flat-embeddings", "flat-rag", "hierarchical-embeddings", "hierarchical-rag", "all"],
        help="Entry point for the API",
    )
    parser.add_argument("--experiment_name", type=str, default="graph-rag-evaluation", help="Experiment name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_samples, entry_point, experiment_name = args.num_samples, args.entry_point, args.experiment_name

    if entry_point == "all":
        entry_points = ["flat-embeddings", "flat-rag", "hierarchical-embeddings", "hierarchical-rag"]
    else:
        entry_points = [entry_point]

    df_test = load_test_data(num_samples)
    df_naf = get_df_naf()
    ground_truth = get_all_levels(df_test, df_naf, col="nace2025")

    api_inputs = {"queries": list(df_test["libelle"].values)}

    for entry_point in entry_points:
        logger.info(f"Evaluating {entry_point} with {num_samples} samples...")
        try:
            evaluate(df_naf, experiment_name, api_inputs, entry_point, num_samples)
        except Exception as e:
            print(f"Error evaluating {entry_point}: {e}")
            traceback.print_exc()
