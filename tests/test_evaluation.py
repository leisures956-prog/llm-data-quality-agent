from app.pipeline.evaluation import (
    evaluate_documents,
)


def test_evaluation_metrics():

    documents = [
        {
            "id": "a",
            "decision": "keep",
            "human_decision": "keep",
        },
        {
            "id": "b",
            "decision": "keep",
            "human_decision": "reject",
        },
        {
            "id": "c",
            "decision": "reject",
            "human_decision": "keep",
        },
        {
            "id": "d",
            "decision": "reject",
            "human_decision": "reject",
        },
    ]

    result = evaluate_documents(documents)
    metrics = result["strict_keep_metrics"]

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert result["false_rejection_count"] == 1
    assert result["false_acceptance_count"] == 1


def test_review_is_counted_separately():

    documents = [
        {
            "id": "review_keep",
            "decision": "review",
            "human_decision": "keep",
        },
        {
            "id": "review_reject",
            "decision": "review",
            "human_decision": "reject",
        },
    ]

    result = evaluate_documents(documents)

    assert result["review_count"] == 2
    assert result["review_keep_count"] == 1
    assert result["review_reject_count"] == 1


def test_metrics_are_grouped_by_language():

    documents = [
        {
            "id": "zh_doc",
            "language": "zh",
            "topic": "technology",
            "decision": "keep",
            "human_decision": "keep",
        },
        {
            "id": "en_doc",
            "language": "en",
            "topic": "technology",
            "decision": "reject",
            "human_decision": "reject",
        },
    ]

    result = evaluate_documents(documents)

    assert "zh" in result["by_language"]
    assert "en" in result["by_language"]
    assert "technology" in result["by_topic"]