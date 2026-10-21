from tools.run_benchmark_regression_gate import (
    DEFAULT_POLICY_PATH,
    evaluate_regression,
    load_json,
    load_jsonl,
    score_feature_fixture,
)


def basic_policy():
    return {
        "version": "test_gate_v1",
        "expected_rule_version": "rules_v1",
        "max_added_ids": 0,
        "max_removed_ids": 0,
        "max_decision_changes": 0,
        "max_score_drop_per_document": 10,
        "max_average_score_drop": 2,
        "max_review_rate": 0.2,
        "allowlisted_changes": [],
    }


def test_fixture_matches_current_quality_rules():
    policy = load_json(DEFAULT_POLICY_PATH)
    baseline = load_jsonl(
        policy["fixture_baseline_path"]
    )
    candidate = score_feature_fixture(
        policy["fixture_features_path"],
        policy["rules_path"],
    )

    report = evaluate_regression(
        baseline,
        candidate,
        policy,
        check_expected=True,
    )

    assert report["status"] == "passed"
    assert report["violations"] == []


def test_gate_rejects_unallowlisted_decision_change():
    old_rows = [
        {
            "id": "doc_001",
            "production_decision": "keep",
            "calibrated_quality_score": 75,
            "quality_rule_version": "rules_v1",
        }
    ]
    new_rows = [
        {
            "id": "doc_001",
            "production_decision": "reject",
            "calibrated_quality_score": 50,
            "quality_rule_version": "rules_v1",
        }
    ]

    report = evaluate_regression(
        old_rows,
        new_rows,
        basic_policy(),
    )

    assert report["status"] == "failed"
    assert any(
        item["type"] == "decision_changes"
        for item in report["violations"]
    )


def test_gate_accepts_explicit_decision_allowlist():
    policy = basic_policy()
    policy["allowlisted_changes"] = [
        {
            "id": "doc_001",
            "after_decision": "reject",
            "max_score_drop": 30,
        }
    ]

    old_rows = [
        {
            "id": "doc_001",
            "production_decision": "keep",
            "calibrated_quality_score": 75,
            "quality_rule_version": "rules_v1",
        }
    ]
    new_rows = [
        {
            "id": "doc_001",
            "production_decision": "reject",
            "calibrated_quality_score": 50,
            "quality_rule_version": "rules_v1",
        }
    ]

    report = evaluate_regression(
        old_rows,
        new_rows,
        policy,
    )

    assert report["status"] == "passed"


def test_gate_rejects_large_score_drop():
    old_rows = [
        {
            "id": "doc_001",
            "production_decision": "keep",
            "calibrated_quality_score": 90,
            "quality_rule_version": "rules_v1",
        }
    ]
    new_rows = [
        {
            "id": "doc_001",
            "production_decision": "keep",
            "calibrated_quality_score": 70,
            "quality_rule_version": "rules_v1",
        }
    ]

    report = evaluate_regression(
        old_rows,
        new_rows,
        basic_policy(),
    )

    assert report["status"] == "failed"
    assert any(
        item["type"] == "score_drop_per_document"
        for item in report["violations"]
    )


def test_gate_rejects_review_rate_over_budget():
    old_rows = [
        {
            "id": "doc_001",
            "production_decision": "keep",
            "calibrated_quality_score": 75,
            "quality_rule_version": "rules_v1",
        },
        {
            "id": "doc_002",
            "production_decision": "keep",
            "calibrated_quality_score": 75,
            "quality_rule_version": "rules_v1",
        },
    ]
    new_rows = [
        {
            "id": "doc_001",
            "production_decision": "review",
            "calibrated_quality_score": 45,
            "quality_rule_version": "rules_v1",
        },
        {
            "id": "doc_002",
            "production_decision": "review",
            "calibrated_quality_score": 45,
            "quality_rule_version": "rules_v1",
        },
    ]

    report = evaluate_regression(
        old_rows,
        new_rows,
        basic_policy(),
    )

    assert report["status"] == "failed"
    assert any(
        item["type"] == "review_rate"
        for item in report["violations"]
    )
