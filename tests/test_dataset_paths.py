from app.pipeline.dataset_paths import (
    DEFAULT_DATASET_PATH,
    get_dataset_path,
)
from pathlib import Path

def test_default_dataset_path():
    assert (
        get_dataset_path()
        == DEFAULT_DATASET_PATH
    )


def test_custom_dataset_path(monkeypatch):
    monkeypatch.setenv(
        "DATASET_PATH",
        "data/raw/benchmark_v1.jsonl",
    )

    assert get_dataset_path() == Path(
        "data/raw/benchmark_v1.jsonl"
    )