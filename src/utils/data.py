import os

import s3fs


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
