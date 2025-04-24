"""
This run can be run directly using:
        uv run evaluate.py

Arguments can be passed using the command line, see --help and parse_args() for more details.

"""

import asyncio
import logging
import os
from typing import List

import httpx
import mlflow
import pandas as pd

from evaluation.args_parser import parse_args
from evaluation.data import get_all_levels, load_test_data, process_response, save_predictions
from utils.data import get_df_naf
from utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

API_URL = "https://codification-ape-graph-rag-api.lab.sspcloud.fr"


async def evaluate_method(
    client: httpx.AsyncClient,
    method: str,
    queries: List[str],
    df_naf: pd.DataFrame,
) -> pd.DataFrame:
    try:
        logger.info(f"🚀 Starting evaluation for '{method}'")
        response = await client.post(
            f"{API_URL}/{method}/batch",
            json={"queries": queries},
            timeout=3600,
        )
        response.raise_for_status()

        preds = process_response(response.json())
        preds_levels = get_all_levels(preds, df_naf, "code_ape")
        save_predictions(preds, method)

        logger.info(f"✅ Finished evaluation for '{method}'")
        return preds_levels

    except Exception as e:
        logger.error(f"❌ Error during '{method}': {e}")
        return pd.DataFrame()


async def evaluate_all(methods: List[str], queries: List[str], df_naf: pd.DataFrame) -> List[pd.DataFrame]:
    async with httpx.AsyncClient() as client:
        tasks = [evaluate_method(client, method, queries, df_naf) for method in methods]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    assert "MLFLOW_TRACKING_URI" in os.environ, "Please set the MLFLOW_TRACKING_URI environment variable."

    args = parse_args()

    methods = (
        ["flat-embeddings", "flat-rag", "hierarchical-embeddings", "hierarchical-rag"]
        if args.entry_point == "all"
        else [args.entry_point]
    )

    df_test = load_test_data(args.num_samples)
    df_naf = get_df_naf()
    ground_truth = get_all_levels(df_test, df_naf, col="nace2025")
    queries = df_test["libelle"].tolist()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name)

    logger.info(f"Evaluating {len(methods)} method(s) with {args.num_samples} samples...")

    with mlflow.start_run():
        if args.entry_point == "all":
            results = asyncio.run(evaluate_all(methods, queries, df_naf))
        else:

            async def eval_single():
                async with httpx.AsyncClient() as client:
                    return await evaluate_method(client, methods[0], queries, df_naf)

            results = [asyncio.run(eval_single())]

    logger.info(results)

    # try:
    #     evaluate(df_naf, experiment_name, api_inputs, entry_point)
    # except Exception as e:
    #     print(f"Error evaluating {entry_point}: {e}")

    #  mlflow.log_param("entry_point", entry_point)
    #     accs = (preds_levels == ground_truth).mean()
    #     for lvl, score in enumerate(accs, 1):
    #         mlflow.log_metric(f"accuracy_lvl_{lvl}", score)
