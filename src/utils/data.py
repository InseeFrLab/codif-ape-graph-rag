import logging
import os

import pandas as pd
import s3fs

logger = logging.getLogger(__name__)


def get_file_system(token=None) -> s3fs.S3FileSystem:
    """
    Creates and returns an S3 file system instance using the s3fs library.

    Parameters:
    -----------
    token : str, optional
        A temporary security token for session-based authentication. This is optional and
        should be provided when using session-based credentials.

    Returns:
    --------
    s3fs.S3FileSystem
        An instance of the S3 file system configured with the specified endpoint and
        credentials, ready to interact with S3-compatible storage.

    """

    options = {
        "client_kwargs": {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    }

    if token is not None:
        options["token"] = token

    return s3fs.S3FileSystem(**options)


def remove_nodes_one_child(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes intermediate parent nodes with exactly one child from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the nodes to process.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the intermediate parent nodes removed.

    """
    # Need to do it level by level
    for level in [3, 2]:
        parent_counts = df["PARENT_CODE"].value_counts()
        parents_one_child = parent_counts[parent_counts == 1].index
        parents_one_child = [code for code in parents_one_child if len(code.replace(".", "")) == level]

        parent_rows = df[df["CODE"].isin(parents_one_child)].set_index("CODE")

        parent_to_grandpa_code = parent_rows["PARENT_CODE"].to_dict()
        raw_parent_ids = parent_rows["PARENT_ID"].to_dict()

        parent_to_grandpa_id = {k.replace(".", ""): v.replace(".", "") for k, v in raw_parent_ids.items()}

        df.loc[:, "PARENT_CODE"] = df["PARENT_CODE"].replace(parent_to_grandpa_code)
        df.loc[:, "PARENT_ID"] = df["PARENT_ID"].replace(parent_to_grandpa_id)

        df = df[~df["CODE"].isin(parents_one_child)]

        for code in parents_one_child:
            logger.info(f"🚮 Removed intermediate parent node: {code}")

    return df


def load_notices(parquet_path: str, columns: list) -> pd.DataFrame:
    logger.info("📄 Loading Parquet data from: %s", parquet_path)
    fs = get_file_system()
    df = pd.read_parquet(parquet_path, filesystem=fs)
    df = remove_nodes_one_child(df)
    return df[columns]
