from app.pipeline.calibrated_scoring import (
    calibrated_score_document,
    calibrated_score_documents,
)


def test_near_duplicate_is_not_penalized_as_low_quality():
    document = {
        "id": "similar",
        "clean_text_chars": 60,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2,
        "title_body_overlap": 1.0,
        "link_ratio": 0.0,
        "is_duplicate": False,
        "is_near_duplicate": True,
    }

    result = calibrated_score_document(document)

    assert result["calibrated_quality_score"] == 60
    assert result["content_decision"] == "review"
    assert result["duplicate_status"] == "near_duplicate"
    assert result["calibrated_decision"] == "review"


def test_short_informative_text_goes_to_review():
    document = {
        "id": "short_good",
        "clean_text_chars": 29,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2,
        "title_body_overlap": 1.0,
        "link_ratio": 0.0,
        "is_duplicate": False,
        "is_near_duplicate": False,
    }

    result = calibrated_score_document(document)

    assert result["calibrated_quality_score"] == 40
    assert result["content_decision"] == "review"
    assert result["calibrated_decision"] == "review"


def test_low_quality_document_is_rejected():
    document = {
        "id": "bad",
        "clean_text_chars": 5,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.8,
        "title_body_overlap": 0.0,
        "link_ratio": 0.0,
        "is_duplicate": False,
        "is_near_duplicate": False,
    }

    result = calibrated_score_document(document)

    assert result["calibrated_decision"] == "reject"


def test_calibrated_content_rich_page_overcomes_link_penalty():
    document = {
        "id": "mdn_like",
        "clean_text_chars": 7336,
        "paragraph_count": 34,
        "sentence_count": 70,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.5552,
        "title_body_overlap": 1.0,
        "link_ratio": 0.4848,
        "is_duplicate": False,
        "is_near_duplicate": False,
    }

    result = calibrated_score_document(document)

    assert result["content_rich"] is True
    assert result["navigation_like"] is False
    assert result["calibrated_decision"] == "keep"
    assert result["calibrated_quality_score"] >= 70


def test_calibrated_navigation_page_is_rejected():
    document = {
        "id": "toc_like",
        "clean_text_chars": 12054,
        "paragraph_count": 8,
        "sentence_count": 972,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2048,
        "title_body_overlap": 1.0,
        "link_ratio": 0.9769,
        "is_duplicate": False,
        "is_near_duplicate": False,
    }

    result = calibrated_score_document(document)

    assert result["navigation_like"] is True
    assert result["content_decision"] == "reject"
    assert result["calibrated_decision"] == "reject"


def test_calibrated_small_index_page_is_rejected():
    document = {
        "id": "small_index",
        "clean_text_chars": 1234,
        "paragraph_count": 6,
        "sentence_count": 8,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.9824,
        "title_body_overlap": 1.0,
        "link_ratio": 0.4895,
        "is_duplicate": False,
        "is_near_duplicate": False,
    }

    result = calibrated_score_document(document)

    assert result["navigation_like"] is True
    assert result["calibrated_decision"] == "reject"


def test_calibrated_blog_aggregation_page_is_rejected():
    document = {
        "id": "blog_home",
        "title": "MDN Blog",
        "clean_text_chars": 12824,
        "paragraph_count": 81,
        "sentence_count": 128,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2,
        "title_body_overlap": 1.0,
        "link_ratio": 0.2998,
        "is_duplicate": False,
        "is_near_duplicate": False,
    }

    result = calibrated_score_document(document)

    assert result["content_rich"] is True
    assert result["aggregation_like"] is True
    assert result["calibrated_decision"] == "reject"
    assert any(
        "聚合页" in reason
        for reason in result["calibrated_reasons"]
    )


def test_review_budget_limits_batch_review_queue():
    documents = [
        {
            "id": f"borderline_{index}",
            "clean_text_chars": 100,
            "paragraph_count": 0,
            "sentence_count": 0,
            "repeated_line_ratio": 0.0,
            "removed_text_ratio": 0.2,
            "title_body_overlap": 1.0,
            "link_ratio": 0.5,
            "is_duplicate": False,
            "is_near_duplicate": False,
        }
        for index in range(5)
    ]

    results = calibrated_score_documents(documents)

    assert sum(
        item["calibrated_decision"] == "review"
        for item in results
    ) == 1
    assert sum(
        item["policy_action"]
        == "review_budget_auto_keep"
        for item in results
    ) == 4
