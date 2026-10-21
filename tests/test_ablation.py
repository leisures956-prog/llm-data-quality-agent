from app.pipeline.ablation import (
    build_rule_variants,
    evaluate_variant,
)


def test_rule_variants_are_created():
    variants = build_rule_variants()

    assert "baseline_v1" in variants
    assert "tolerant_review_v2" in variants
    assert "strict_review_v2" in variants
    assert (
        "without_duplicate_review"
        in variants
    )


def test_strict_and_duplicate_ablation_change_result():
    document = {
        "id": "short_doc",
        "clean_text_chars": 29,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2,
        "title_body_overlap": 1.0,
        "link_ratio": 0.0,
        "is_duplicate": False,
        "is_near_duplicate": False,
        "human_decision": "keep",
    }

    variants = build_rule_variants()

    baseline_summary, baseline_results = (
        evaluate_variant(
            [document],
            "baseline_v1",
            variants["baseline_v1"],
        )
    )

    strict_summary, strict_results = (
        evaluate_variant(
            [document],
            "strict_review_v2",
            variants["strict_review_v2"],
        )
    )

    assert (
        baseline_results[0][
            "calibrated_decision"
        ]
        == "review"
    )

    assert (
        strict_results[0][
            "calibrated_decision"
        ]
        == "reject"
    )

    assert (
        strict_summary["false_rejection_count"]
        == 1
    ) 