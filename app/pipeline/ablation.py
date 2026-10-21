import copy
from collections import Counter

from app.pipeline.calibrated_scoring import (
    calibrated_score_document,
)
from app.pipeline.rule_config import (
    load_quality_rules,
)


def build_rule_variants(
    base_rules: dict | None = None,
) -> dict[str, dict]:
    if base_rules is None:
        base_rules = load_quality_rules()

    baseline = copy.deepcopy(base_rules)
    baseline["version"] = (
        "quality_rules_v1_baseline"
    )

    tolerant = copy.deepcopy(base_rules)
    tolerant["version"] = (
        "quality_rules_v2_tolerant_review"
    )
    tolerant["decision"][
        "review_min_score"
    ] = 15

    strict = copy.deepcopy(base_rules)
    strict["version"] = (
        "quality_rules_v2_strict_review"
    )
    strict["decision"][
        "review_min_score"
    ] = 41

    no_duplicate_review = copy.deepcopy(
        base_rules
    )
    no_duplicate_review["version"] = (
        "ablation_without_duplicate_review"
    )
    no_duplicate_review["near_duplicate"][
        "non_reject_decision"
    ] = "keep"

    return {
        "baseline_v1": baseline,
        "tolerant_review_v2": tolerant,
        "strict_review_v2": strict,
        "without_duplicate_review": (
            no_duplicate_review
        ),
    }


def _rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator,
        4,
    )


def evaluate_variant(
    documents: list[dict],
    variant_name: str,
    rules: dict,
) -> tuple[dict, list[dict]]:
    results = [
        calibrated_score_document(
            document,
            rules=rules,
        )
        for document in documents
    ]

    decision_counts = Counter(
        item["calibrated_decision"]
        for item in results
    )

    labeled = [
        item
        for item in results
        if item.get("human_decision")
        in {"keep", "reject"}
    ]

    true_positive = sum(
        item["calibrated_decision"] == "keep"
        and item["human_decision"] == "keep"
        for item in labeled
    )

    false_positive = sum(
        item["calibrated_decision"] == "keep"
        and item["human_decision"] == "reject"
        for item in labeled
    )

    false_negative = sum(
        item["calibrated_decision"] != "keep"
        and item["human_decision"] == "keep"
        for item in labeled
    )

    true_negative = sum(
        item["calibrated_decision"] != "keep"
        and item["human_decision"] == "reject"
        for item in labeled
    )

    precision = _rate(
        true_positive,
        true_positive + false_positive,
    )

    recall = _rate(
        true_positive,
        true_positive + false_negative,
    )

    f1 = _rate(
        2 * precision * recall,
        precision + recall,
    )

    false_rejection_count = sum(
        item["calibrated_decision"] == "reject"
        and item["human_decision"] == "keep"
        for item in labeled
    )

    false_acceptance_count = sum(
        item["calibrated_decision"] == "keep"
        and item["human_decision"] == "reject"
        for item in labeled
    )

    review_keep_count = sum(
        item["calibrated_decision"] == "review"
        and item["human_decision"] == "keep"
        for item in labeled
    )

    average_score = _rate(
        sum(
            item["calibrated_quality_score"]
            for item in results
        ),
        len(results),
    )

    summary = {
        "variant_name": variant_name,
        "rule_version": rules["version"],
        "total": len(results),
        "decision_counts": dict(
            decision_counts
        ),
        "average_score": average_score,
        "false_rejection_count": (
            false_rejection_count
        ),
        "false_acceptance_count": (
            false_acceptance_count
        ),
        "review_keep_count": review_keep_count,
        "strict_keep_metrics": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }

    return summary, results