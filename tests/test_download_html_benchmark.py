from tools.download_html_benchmark import (
    DEFAULT_ERROR_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH,
    get_download_paths,
)


def test_download_paths_keep_v2_defaults(monkeypatch):
    for name in (
        "HTML_MANIFEST_PATH",
        "HTML_OUTPUT_PATH",
        "HTML_ERROR_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    paths = get_download_paths()

    assert paths["manifest"] == DEFAULT_MANIFEST_PATH
    assert paths["output"] == DEFAULT_OUTPUT_PATH
    assert paths["errors"] == DEFAULT_ERROR_PATH


def test_download_paths_can_be_versioned(monkeypatch):
    monkeypatch.setenv(
        "HTML_MANIFEST_PATH",
        "data/raw/html_urls_v3.jsonl",
    )
    monkeypatch.setenv(
        "HTML_OUTPUT_PATH",
        "data/raw/html_benchmark_v3.jsonl",
    )
    monkeypatch.setenv(
        "HTML_ERROR_PATH",
        "data/raw/html_benchmark_v3_errors.jsonl",
    )

    paths = get_download_paths()

    assert paths["manifest"].as_posix().endswith(
        "data/raw/html_urls_v3.jsonl"
    )
    assert paths["output"].as_posix().endswith(
        "data/raw/html_benchmark_v3.jsonl"
    )
    assert paths["errors"].as_posix().endswith(
        "data/raw/html_benchmark_v3_errors.jsonl"
    )
