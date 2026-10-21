from app.pipeline.sampling import (
    assign_sampling_weights,
    summarize_sampling,
)


def test_rejected_document_has_zero_weight():
    documents = [
        {
            "id": "bad",
            "language": "zh",
            "topic": "news",
            "quality_bucket": "reject",
        }
    ]

    result = assign_sampling_weights(documents)[0]

    assert result["sampling_weight"] == 0.0
    assert result["sampling_status"] == "excluded"


def test_underrepresented_group_gets_higher_factor():
    documents = [
        {
            "id": "zh_1",
            "language": "zh",
            "topic": "technology",
            "quality_bucket": "keep",
        },
        {
            "id": "zh_2",
            "language": "zh",
            "topic": "technology",
            "quality_bucket": "keep",
        },
        {
            "id": "en_1",
            "language": "en",
            "topic": "technology",
            "quality_bucket": "keep",
        },
    ]

    results = assign_sampling_weights(documents)

    zh_weight = results[0]["sampling_weight"]
    en_weight = results[2]["sampling_weight"]

    assert en_weight > zh_weight


def test_sampling_summary_counts_documents():
    documents = [
        {
            "id": "keep",
            "language": "zh",
            "topic": "technology",
            "quality_bucket": "keep",
        },
        {
            "id": "reject",
            "language": "zh",
            "topic": "news",
            "quality_bucket": "reject",
        },
    ]

    results = assign_sampling_weights(documents)
    summary = summarize_sampling(results)

    assert summary["total"] == 2
    assert summary["eligible_count"] == 1
    assert summary["excluded_count"] == 1