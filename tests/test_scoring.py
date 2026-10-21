from app.pipeline.scoring import score_document


def test_high_quality_document_is_kept():
    document = {
        "id": "good",
        "clean_text_chars": 180,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2,
        "title_body_overlap": 0.8,
        "link_ratio": 0.0,
        "is_near_duplicate": False,
    }

    result = score_document(document)

    assert result["quality_score"] >= 70
    assert result["decision"] == "keep"


def test_short_document_is_rejected():
    document = {
        "id": "short",
        "clean_text_chars": 5,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2,
        "title_body_overlap": 0.0,
        "link_ratio": 0.0,
        "is_near_duplicate": False,
    }

    result = score_document(document)

    assert result["decision"] == "reject"


def test_near_duplicate_requires_review():
    document = {
        "id": "similar",
        "clean_text_chars": 60,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2,
        "title_body_overlap": 1.0,
        "link_ratio": 0.0,
        "is_near_duplicate": True,
    }

    result = score_document(document)

    assert result["decision"] == "review"
    assert any(
        "相似" in reason
        for reason in result["reasons"]
    )


def test_content_rich_link_heavy_document_is_kept():
    document = {
        "id": "mdn_like",
        "clean_text_chars": 7336,
        "paragraph_count": 34,
        "sentence_count": 70,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.5552,
        "title_body_overlap": 1.0,
        "link_ratio": 0.4848,
        "is_near_duplicate": False,
    }

    result = score_document(document)

    assert result["content_rich"] is True
    assert result["navigation_like"] is False
    assert result["decision"] == "keep"
    assert any(
        "内容密度" in reason
        for reason in result["reasons"]
    )


def test_navigation_like_page_is_rejected():
    document = {
        "id": "toc_like",
        "clean_text_chars": 12054,
        "paragraph_count": 8,
        "sentence_count": 972,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2048,
        "title_body_overlap": 1.0,
        "link_ratio": 0.9769,
        "is_near_duplicate": False,
    }

    result = score_document(document)

    assert result["navigation_like"] is True
    assert result["decision"] == "reject"


def test_small_index_page_is_rejected():
    document = {
        "id": "small_index",
        "clean_text_chars": 1234,
        "paragraph_count": 6,
        "sentence_count": 8,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.9824,
        "title_body_overlap": 1.0,
        "link_ratio": 0.4895,
        "is_near_duplicate": False,
    }

    result = score_document(document)

    assert result["navigation_like"] is True
    assert result["decision"] == "reject"


def test_blog_aggregation_page_is_rejected():
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
        "is_near_duplicate": False,
    }

    result = score_document(document)

    assert result["content_rich"] is True
    assert result["aggregation_like"] is True
    assert result["decision"] == "reject"
    assert any(
        "聚合页" in reason
        for reason in result["reasons"]
    )
