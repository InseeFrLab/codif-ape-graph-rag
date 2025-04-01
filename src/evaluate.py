import os
import sys

import httpx
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("codif-ape-train/src/"))  # Adjust the path if needed
# sys.path.append(os.path.abspath("../codif-ape-train/src"))  # Adjust the path if needed

from evaluators import Evaluator

from constants.api_url import API_BASE_URL
from utils.data import get_file_system

NUM_SAMPLES = 1000


def process_response(response):
    """
    Get json response from the API and return predictions and probabilites Numpy arrays.
    """
    preds = pd.DataFrame(response)
    preds["code_ape"] = preds["code_ape"].str.replace(".", "", regex=False)
    probs = np.ones_like(preds)  # placeholder

    return preds.values, probs


def evaluate(entry_point: str):
    fs = get_file_system()
    df_test = pd.read_parquet(
        "projet-ape/model_comparison_splits/sirene_4_NAF2025_20241027/df_test_raw.parquet",
        filesystem=fs,
    )

    api_inputs = {"queries": list(df_test["libelle"].values)[:NUM_SAMPLES]}

    response = httpx.post(f"{API_BASE_URL}/{entry_point}/batch", json=api_inputs, timeout=600)
    response.raise_for_status()
    results = response.json()

    preds, probs = process_response(results)

    df_res = Evaluator.get_aggregated_preds(
        df=df_test.head(NUM_SAMPLES), predictions=preds, probabilities=probs, Y="nace2025", int_to_str=False, revision="NAF2025"
    )

    file_path = f"projet-ape/model_comparison_splits/sirene_4_NAF2025_20241027/df_test_raw_pred_{entry_point}.parquet"
    with fs.open(file_path, "wb") as f:
        df_res.to_parquet(f)

    return df_res


if __name__ == "__main__":
    evaluate("flat-embeddings")
