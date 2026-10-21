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


def test_experiment_scope_preserves_unverified_source():
    documents = [
        {
            "id": "experiment_doc",
            "language": "en",
            "topic": "technology",
            "quality_bucket": "keep",
            "source_registered": True,
            "license_status": "to_verify",
            "release_eligible": False,
        }
    ]

    result = assign_sampling_weights(
        documents,
        sampling_scope="experiment",
    )[0]

    assert result["sampling_status"] == "eligible"
    assert result["sampling_weight"] > 0
    assert result["policy_eligible"] is True


def test_release_scope_excludes_unverified_source():
    documents = [
        {
            "id": "release_doc",
            "language": "en",
            "topic": "technology",
            "quality_bucket": "keep",
            "source_registered": True,
            "license_status": "to_verify",
            "release_eligible": False,
        }
    ]

    result = assign_sampling_weights(
        documents,
        sampling_scope="release",
    )[0]

    assert result["sampling_status"] == "excluded"
    assert result["sampling_weight"] == 0.0
    assert result["sampling_exclusion_reason"] == (
        "license_not_verified"
    )


def test_release_scope_allows_verified_source():
    documents = [
        {
            "id": "verified_doc",
            "language": "en",
            "topic": "technology",
            "quality_bucket": "keep",
            "source_registered": True,
            "license_status": "verified",
            "release_eligible": True,
        }
    ]

    result = assign_sampling_weights(
        documents,
        sampling_scope="release",
    )[0]

    assert result["sampling_status"] == "eligible"
    assert result["sampling_weight"] > 0


def test_sampling_summary_reports_policy_exclusions():
    documents = [
        {
            "id": "pending",
            "language": "en",
            "topic": "technology",
            "quality_bucket": "keep",
            "source_registered": True,
            "license_status": "to_verify",
            "release_eligible": False,
        },
        {
            "id": "rejected",
            "language": "en",
            "topic": "news",
            "quality_bucket": "reject",
            "source_registered": True,
            "license_status": "to_verify",
            "release_eligible": False,
        },
    ]

    results = assign_sampling_weights(
        documents,
        sampling_scope="release",
    )
    summary = summarize_sampling(
        results,
        sampling_scope="release",
    )

    assert summary["sampling_scope"] == "release"
    assert summary["quality_excluded_count"] == 1
    assert summary["policy_eligible_count"] == 0
    assert summary["policy_excluded_count"] == 1
    assert summary["policy_exclusion_counts"] == {
        "license_not_verified": 1
    }
    assert summary["eligible_count"] == 0
