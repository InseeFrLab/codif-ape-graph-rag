from typing import List

import pandas as pd

from constants.paths import PRED_TEST_DATA_PATH, TEST_DATA_PATH
from utils.data import get_file_system


def load_test_data(num_samples: int) -> pd.DataFrame:
    fs = get_file_system()
    return pd.read_parquet(TEST_DATA_PATH, filesystem=fs, columns=["nace2025", "libelle"]).head(num_samples)


def get_all_levels(df: pd.DataFrame, df_naf: pd.DataFrame, col: str) -> pd.DataFrame:
    # Invalid NAF2025 code are replaced by NaN here (with how="left")
    return df[[col]].merge(df_naf, how="left", left_on=col, right_on="APE_NIV5")[df_naf.columns.drop(["LIB_NIV5"])]


def process_response(raw_response: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_response)
    df["code_ape"] = df["code_ape"].str.replace(".", "", regex=False)
    return df


def save_predictions(df: pd.DataFrame, entry_point: str) -> None:
    fs = get_file_system()
    path = f"{PRED_TEST_DATA_PATH}_{entry_point}.parquet"
    with fs.open(path, "wb") as f:
        df.to_parquet(f)
