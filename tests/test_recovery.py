import pytest

from app.pipeline.recovery import (
    apply_manual_labels,
    summarize_recovery,
)


def test_false_rejection_is_recovered():
    documents = [
        {
            "id": "doc_008",
            "decision": "reject",
            "quality_score": 30,
        }
    ]
    labels = [
        {
            "id": "doc_008",
            "human_decision": "keep",
            "is_false_rejection": True,
            "action": "keep_short_informative",
            "reason": "内容虽短但有价值",
        }
    ]

    results = apply_manual_labels(documents, labels)
    summary = summarize_recovery(results)

    assert results[0]["final_decision"] == "keep"
    assert results[0]["recovery_status"] == "recovered"
    assert summary["false_rejection_count"] == 1
    assert summary["recovered_count"] == 1
    assert summary["recovery_rate"] == 100.0


def test_human_rejection_remains_rejected():
    documents = [{"id": "bad", "decision": "reject"}]
    labels = [
        {
            "id": "bad",
            "human_decision": "reject",
            "is_false_rejection": False,
            "action": "remove_short",
            "reason": "内容无信息量",
        }
    ]

    result = apply_manual_labels(documents, labels)[0]

    assert result["final_decision"] == "reject"
    assert result["recovery_status"] == "confirmed"


def test_missing_manual_label_fails_closed():
    documents = [
        {"id": "doc_a", "decision": "keep"},
        {"id": "doc_b", "decision": "reject"},
    ]
    labels = [
        {
            "id": "doc_a",
            "human_decision": "keep",
        }
    ]

    with pytest.raises(ValueError, match="缺少人工标签"):
        apply_manual_labels(documents, labels)
