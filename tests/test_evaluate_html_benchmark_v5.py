import pytest

from tools.evaluate_html_benchmark_v5 import (
    evaluate_html_benchmark_v5,
)


def _decision(item_id, decision, title=None):
    return {
        "id": item_id,
        "title": title or item_id,
        "production_decision": decision,
        "language": "en",
        "benchmark_category": "technical_content",
    }


def _label(item_id, human_decision, split="dev", error_type=None):
    return {
        "id": item_id,
        "split": split,
        "human_decision": human_decision,
        "error_type": error_type,
        "review_reason": "test reason",
        "annotator_confidence": 0.9,
        "reviewed": True,
    }


def test_v5_reports_hard_errors_review_and_coverage():
    decisions = [
        _decision("a", "reject"),
        _decision("b", "keep"),
        _decision("c", "review"),
        _decision("d", "reject"),
    ]
    labels = [
        _label("a", "keep", error_type="too_short"),
        _label("b", "reject", error_type="navigation_page"),
        _label("c", "keep", error_type="aggregation_page"),
        _label("d", "reject"),
    ]

    report = evaluate_html_benchmark_v5(decisions, labels)

    assert report["total"] == 4
    assert report["hard_false_rejection_count"] == 1
    assert report["hard_false_rejection_rate"] == 0.5
    assert report["false_acceptance_count"] == 1
    assert report["false_acceptance_rate"] == 0.5
    assert report["review_count"] == 1
    assert report["review_rate"] == 0.25
    assert report["automatic_decision_count"] == 3
    assert report["automatic_decision_coverage"] == 0.75
    assert report["decision_cross_table"]["review"]["keep"] == 1
    assert report["bad_case_counts"]["too_short"] == 1
    assert len(report["error_cases"]) == 2
    assert len(report["review_cases"]) == 1


def test_v5_excludes_blind_test_labels():
    decisions = [
        _decision("dev", "keep"),
        _decision("blind", "reject"),
    ]
    labels = [
        _label("dev", "keep"),
        {
            "id": "blind",
            "split": "blind_test",
            "human_decision": "",
            "reviewed": False,
        },
    ]

    report = evaluate_html_benchmark_v5(decisions, labels)

    assert report["total"] == 1
    assert report["machine_decision_counts"] == {"keep": 1}


def test_v5_rejects_unreviewed_development_label():
    decisions = [_decision("a", "keep")]
    labels = [_label("a", "keep")]
    labels[0]["reviewed"] = False

    with pytest.raises(ValueError, match="reviewed=true"):
        evaluate_html_benchmark_v5(decisions, labels)
