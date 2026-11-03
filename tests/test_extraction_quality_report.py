from tools.build_extraction_quality_report import (
    build_extraction_quality_report,
)


def test_extraction_quality_report_summarizes_flags_and_groups():
    features = [
        {
            "id": "doc_001",
            "title": "正文页",
            "source": "docs.example.com",
            "benchmark_category": "technical_content",
            "language": "en",
            "clean_text_chars": 500,
            "paragraph_count": 5,
            "sentence_count": 20,
            "link_ratio": 0.1,
            "removed_text_ratio": 0.1,
        },
        {
            "id": "doc_002",
            "title": "目录页",
            "source": "docs.example.com",
            "benchmark_category": "index_page",
            "language": "en",
            "clean_text_chars": 80,
            "paragraph_count": 1,
            "sentence_count": 2,
            "link_ratio": 0.8,
            "removed_text_ratio": 0.75,
        },
    ]
    decisions = [
        {
            "id": "doc_001",
            "production_decision": "keep",
        },
        {
            "id": "doc_002",
            "production_decision": "reject",
        },
    ]
    errors = [
        {
            "id": "doc_003",
            "error_type": "HTTPStatusError",
        },
    ]

    report = build_extraction_quality_report(
        features,
        decision_rows=decisions,
        error_rows=errors,
    )

    assert report["summary"]["total"] == 2
    assert report["summary"]["empty_clean_text_count"] == 0
    assert report["summary"]["short_clean_text_count"] == 1
    assert report["summary"][
        "high_removed_text_ratio_count"
    ] == 1
    assert report["summary"]["high_link_ratio_count"] == 1
    assert report["summary"][
        "single_or_no_paragraph_count"
    ] == 1
    assert report["summary"]["decision_counts"] == {
        "keep": 1,
        "reject": 1,
    }
    assert report["download_errors"]["total"] == 1
    assert report["by_category"]["index_page"][
        "flagged_document_count"
    ] == 1
    assert len(report["flagged_documents"]) == 1


def test_extraction_quality_report_tracks_missing_decisions():
    report = build_extraction_quality_report(
        [
            {
                "id": "doc_001",
                "clean_text_chars": 0,
                "paragraph_count": 0,
                "sentence_count": 0,
                "link_ratio": 0.0,
                "removed_text_ratio": 1.0,
            }
        ],
        decision_rows=[
            {
                "id": "doc_missing",
                "decision": "keep",
            }
        ],
    )

    assert report["decision_coverage"][
        "features_without_decision"
    ] == ["doc_001"]
    assert report["decision_coverage"][
        "decisions_without_features"
    ] == ["doc_missing"]
