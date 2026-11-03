from tools.build_collection_quality_report import (
    build_collection_quality_report,
)


def test_collection_report_summarizes_coverage_and_failures():
    manifest = [
        {
            "id": "doc_001",
            "url": "https://docs.example.com/a",
            "source": "docs.example.com",
            "language": "en",
            "benchmark_category": "technical_content",
        },
        {
            "id": "doc_002",
            "url": "https://docs.example.com/b",
            "source": "docs.example.com",
            "language": "en",
            "benchmark_category": "index_page",
        },
        {
            "id": "doc_003",
            "url": "https://zh.example.com/c",
            "source": "zh.example.com",
            "language": "zh",
            "benchmark_category": "technical_content",
        },
    ]
    dataset = [
        {
            "id": "doc_001",
            "source": "docs.example.com",
            "language": "en",
            "benchmark_category": "technical_content",
        },
        {
            "id": "doc_003",
            "source": "zh.example.com",
            "language": "zh",
            "benchmark_category": "technical_content",
        },
    ]
    errors = [
        {
            "id": "doc_002",
            "url": "https://docs.example.com/b",
            "error_type": "HTTPStatusError",
            "error": "403 Forbidden",
        },
    ]

    report = build_collection_quality_report(
        manifest,
        dataset,
        errors,
    )

    assert report["summary"] == {
        "manifest_count": 3,
        "downloaded_count": 2,
        "error_count": 1,
        "success_rate": 0.6667,
        "error_rate": 0.3333,
        "retryable_error_count": 0,
        "non_retryable_error_count": 1,
        "unaccounted_count": 0,
    }
    assert report["coverage_by_language"]["en"] == {
        "expected": 2,
        "downloaded": 1,
        "errors": 1,
        "unaccounted": 0,
        "success_rate": 0.5,
    }
    assert report["error_class_counts"] == {
        "blocked_or_unavailable": 1,
    }
    assert report["id_integrity"][
        "missing_manifest_ids"
    ] == []


def test_collection_report_tracks_unaccounted_and_unknown_ids():
    report = build_collection_quality_report(
        [{"id": "doc_001", "language": "en"}],
        [{"id": "doc_unknown", "language": "en"}],
        [],
    )

    assert report["summary"]["unaccounted_count"] == 1
    assert report["id_integrity"][
        "missing_manifest_ids"
    ] == ["doc_001"]
    assert report["id_integrity"][
        "dataset_unknown_ids"
    ] == ["doc_unknown"]
