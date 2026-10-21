import os
from pathlib import Path


DEFAULT_DATASET_PATH = Path(
    "data/raw/sample.jsonl"
)


def get_dataset_path() -> Path:
    return Path(
        os.environ.get(
            "DATASET_PATH",
            str(DEFAULT_DATASET_PATH),
        )
    )