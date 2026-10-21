from collections import Counter


VALID_HUMAN_DECISIONS = {"keep", "reject"}


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _evaluate_subset(documents: list[dict]) -> dict:
    true_positive = sum(
        item["decision"] == "keep"
        and item["human_decision"] == "keep"
        for item in documents
    )

    false_positive = sum(
        item["decision"] == "keep"
        and item["human_decision"] == "reject"
        for item in documents
    )

    false_negative = sum(
        item["decision"] != "keep"
        and item["human_decision"] == "keep"
        for item in documents
    )

    true_negative = sum(
        item["decision"] != "keep"
        and item["human_decision"] == "reject"
        for item in documents
    )

    precision = _safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = _safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1 = _safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    accuracy = _safe_divide(
        true_positive + true_negative,
        len(documents),
    )

    return {
        "total": len(documents),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_documents(documents: list[dict]) -> dict:
    for item in documents:
        if item.get("human_decision") not in (
            VALID_HUMAN_DECISIONS
        ):
            raise ValueError(
                f"缺少有效人工标签：{item.get('id')}"
            )

    language_groups = {}
    topic_groups = {}

    for item in documents:
        language = item.get(
            "language",
            "unknown",
        )
        topic = item.get(
            "topic",
            "other",
        )

        language_groups.setdefault(
            language,
            [],
        ).append(item)

        topic_groups.setdefault(
            topic,
            [],
        ).append(item)

    confusion_matrix = {}

    for machine_decision in (
        "keep",
        "review",
        "reject",
    ):
        confusion_matrix[machine_decision] = {
            human_decision: sum(
                item["decision"] == machine_decision
                and item["human_decision"]
                == human_decision
                for item in documents
            )
            for human_decision in (
                "keep",
                "reject",
            )
        }

    strict_metrics = _evaluate_subset(documents)

    false_rejection_count = sum(
        item["decision"] == "reject"
        and item["human_decision"] == "keep"
        for item in documents
    )

    false_acceptance_count = sum(
        item["decision"] == "keep"
        and item["human_decision"] == "reject"
        for item in documents
    )

    review_count = sum(
        item["decision"] == "review"
        for item in documents
    )

    review_keep_count = sum(
        item["decision"] == "review"
        and item["human_decision"] == "keep"
        for item in documents
    )

    review_reject_count = sum(
        item["decision"] == "review"
        and item["human_decision"] == "reject"
        for item in documents
    )

    return {
        "total": len(documents),
        "machine_decision_counts": dict(
            Counter(
                item["decision"]
                for item in documents
            )
        ),
        "human_decision_counts": dict(
            Counter(
                item["human_decision"]
                for item in documents
            )
        ),
        "confusion_matrix": confusion_matrix,
        "strict_keep_metrics": strict_metrics,
        "false_rejection_count": (
            false_rejection_count
        ),
        "false_acceptance_count": (
            false_acceptance_count
        ),
        "review_count": review_count,
        "review_keep_count": review_keep_count,
        "review_reject_count": review_reject_count,
        "by_language": {
            language: _evaluate_subset(items)
            for language, items
            in language_groups.items()
        },
        "by_topic": {
            topic: _evaluate_subset(items)
            for topic, items
            in topic_groups.items()
        },
    }