from app.pipeline.dataset_registry import (
    summarize_dataset,
    validate_raw_document,
)


def test_valid_raw_document():
    document = {
        "id": "doc_001",
        "url": "https://example.com",
        "domain": "technology",
        "language": "en",
        "html": (
            "<html><body>"
            "A valid document."
            "</body></html>"
        ),
    }

    errors = validate_raw_document(
        document,
        line_number=1,
    )

    assert errors == []


def test_missing_required_field():
    document = {
        "id": "doc_001",
        "url": "https://example.com",
        "domain": "technology",
        "language": "en",
    }

    errors = validate_raw_document(
        document,
        line_number=1,
    )

    assert any(
        "html" in error
        for error in errors
    )


def test_dataset_summary():
    documents = [
        {
            "id": "doc_001",
            "language": "zh",
            "domain": "technology",
            "html": "<html>内容</html>",
        },
        {
            "id": "doc_002",
            "language": "en",
            "domain": "news",
            "html": "<html>content</html>",
        },
    ]

    summary = summarize_dataset(
        documents
    )

    assert summary["total"] == 2
    assert summary["language_counts"] == {
        "zh": 1,
        "en": 1,
    }
    assert summary["domain_counts"] == {
        "technology": 1,
        "news": 1,
    }