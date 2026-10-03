"""Stage 1 - data loading.

Downloads the UCI SMS Spam Collection (if not already present) and loads it
into a DataFrame. In a production system this stage would pull from a
database, object storage or a message queue instead.
"""

import csv
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def download_dataset(
    url: str = config.DATASET_URL,
    dest_dir: Path = config.RAW_DATA_DIR,
) -> Path:
    """Download and extract the dataset; skip if it already exists."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dest_dir / config.RAW_DATA_PATH.name

    if raw_path.exists():
        logger.info("Dataset already present at %s - skipping download", raw_path)
        return raw_path

    zip_path = dest_dir / config.DATASET_ZIP_PATH.name
    logger.info("Downloading dataset from %s ...", url)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(config.RAW_DATA_PATH.name, dest_dir)
    logger.info("Extracted %s", raw_path)
    return raw_path


def load_dataset(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw TSV file into a DataFrame with proper column names.

    The file was inspected before writing this code: one TAB-separated
    `label<TAB>message` pair per line, no header, UTF-8 encoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run download_dataset() first."
        )
    # QUOTE_NONE is deliberate: two messages in this file start with a double
    # quote, which makes pandas' default parser swallow the FOLLOWING line
    # into a "multiline field" (5,574 -> 5,572 rows). Raw line inspection and
    # the official UCI readme both confirm 5,574 one-message-per-line records,
    # so we disable quote handling entirely.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[config.LABEL_COLUMN, config.MESSAGE_COLUMN],
        encoding=config.DATASET_ENCODING,
        quoting=csv.QUOTE_NONE,
    )
    logger.info(
        "Loaded dataset: %d rows x %d columns from %s", len(df), df.shape[1], path
    )
    logger.info("Columns: %s | dtypes: %s", list(df.columns), df.dtypes.to_dict())
    return df
