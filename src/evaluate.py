import os
import sys

import httpx
import pandas as pd

sys.path.append(os.path.abspath("codif-ape-train/src/"))  # Adjust the path if needed
# sys.path.append(os.path.abspath("../codif-ape-train/src"))  # Adjust the path if needed

from evaluators import Evaluator

from constants.api_url import API_BASE_URL
from utils.data import get_file_system, get_Y


def process_response(response):
    """
    Get json response from the API and return predictions and probabilites Numpy arrays.
    """
    res = pd.DataFrame(response)
    return res


def evaluate(entry_point: str):
    fs = get_file_system()
    df_test = pd.read_parquet(
        "projet-ape/model_comparison_splits/sirene4_20230101_20250211/df_test_raw.parquet",
        filesystem=fs,
    )

    api_inputs = {"queries": list(df_test["libelle"].values)[:10]}

    Y = get_Y(revision="NAF2025")

    response = httpx.post(f"{API_BASE_URL}/{entry_point}/batch", json=api_inputs, timeout=60)
    response.raise_for_status()
    results = response.json()

    preds = process_response(results)

    df_res = Evaluator.get_aggregated_preds(
        df=df_test,
        predictions=preds.code_ape,
        probabilities=None,
        Y=Y,
        int_to_str=False,
    )

    return df_res


if __name__ == "__main__":
    evaluate("flat-embeddings")
