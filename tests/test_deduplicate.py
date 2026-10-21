from app.pipeline.deduplicate import (
    annotate_exact_duplicates,
    normalize_text,
    text_hash,
)


def test_normalize_text():
    text_a = "Data   Quality\nTest"
    text_b = "data quality test"

    assert normalize_text(text_a) == normalize_text(text_b)
    assert text_hash(text_a) == text_hash(text_b)


def test_exact_duplicate_grouping():
    documents = [
        {
            "id": "doc_a",
            "clean_text": "相同的正文内容",
        },
        {
            "id": "doc_b",
            "clean_text": "相同的正文内容",
        },
        {
            "id": "doc_c",
            "clean_text": "另一段正文内容",
        },
    ]

    results = annotate_exact_duplicates(documents)

    duplicate_results = [
        item
        for item in results
        if item["is_duplicate"]
    ]

    assert len(duplicate_results) == 2
    assert (
        duplicate_results[0]["duplicate_group"]
        == duplicate_results[1]["duplicate_group"]
    )

    unique_result = results[2]

    assert unique_result["is_duplicate"] is False
    assert unique_result["duplicate_group"] is None
    

from app.pipeline.deduplicate import (
    select_unique_documents,
)


def test_select_unique_documents():
    documents = [
        {
            "id": "doc_a",
            "duplicate_group": "dup_001",
        },
        {
            "id": "doc_b",
            "duplicate_group": "dup_001",
        },
        {
            "id": "doc_c",
            "duplicate_group": None,
        },
    ]

    results = select_unique_documents(documents)

    result_ids = [
        item["id"]
        for item in results
    ]

    assert result_ids == [
        "doc_a",
        "doc_c",
    ]