import math
from collections import Counter


QUALITY_WEIGHTS = {
    "keep": 1.0,
    "review": 0.5,
    "reject": 0.0,
}


def sampling_group(document: dict) -> str:
    language = document.get("language", "unknown")
    topic = document.get("topic", "other")
    return f"{language}::{topic}"


def assign_sampling_weights(
    documents: list[dict],
) -> list[dict]:
    group_counts = Counter(
        sampling_group(document)
        for document in documents
    )

    results = []

    for document in documents:
        group = sampling_group(document)
        quality = document.get(
            "quality_bucket",
            "review",
        )

        base_weight = QUALITY_WEIGHTS.get(
            quality,
            0.5,
        )

        balance_factor = 1 / math.sqrt(
            group_counts[group]
        )

        sampling_weight = round(
            base_weight * balance_factor,
            4,
        )

        results.append(
            {
                **document,
                "sampling_group": group,
                "base_sampling_weight": base_weight,
                "sampling_balance_factor": round(
                    balance_factor,
                    4,
                ),
                "sampling_weight": sampling_weight,
                "sampling_status": (
                    "eligible"
                    if sampling_weight > 0
                    else "excluded"
                ),
            }
        )

    return results


def summarize_sampling(
    documents: list[dict],
) -> dict:
    group_stats = {}

    for document in documents:
        group = document["sampling_group"]

        if group not in group_stats:
            group_stats[group] = {
                "total": 0,
                "eligible": 0,
                "total_weight": 0.0,
            }

        group_stats[group]["total"] += 1
        group_stats[group]["eligible"] += (
            document["sampling_status"] == "eligible"
        )
        group_stats[group]["total_weight"] += (
            document["sampling_weight"]
        )

    for stats in group_stats.values():
        stats["total_weight"] = round(
            stats["total_weight"],
            4,
        )

    return {
        "total": len(documents),
        "eligible_count": sum(
            item["sampling_status"] == "eligible"
            for item in documents
        ),
        "excluded_count": sum(
            item["sampling_status"] == "excluded"
            for item in documents
        ),
        "total_sampling_weight": round(
            sum(
                item["sampling_weight"]
                for item in documents
            ),
            4,
        ),
        "group_stats": group_stats,
    }