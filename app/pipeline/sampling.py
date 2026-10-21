import math
from collections import Counter


QUALITY_WEIGHTS = {
    "keep": 1.0,
    "review": 0.5,
    "reject": 0.0,
}

SAMPLING_SCOPES = {
    "experiment",
    "release",
}


def sampling_group(document: dict) -> str:
    language = document.get("language", "unknown")
    topic = document.get("topic", "other")
    return f"{language}::{topic}"


def normalize_sampling_scope(
    sampling_scope: str,
) -> str:
    scope = str(
        sampling_scope or "experiment"
    ).strip().casefold()

    if scope not in SAMPLING_SCOPES:
        raise ValueError(
            "不支持的采样范围："
            f"{scope}，可选值为 experiment 或 release"
        )

    return scope


def assess_sampling_policy(
    document: dict,
    sampling_scope: str,
) -> tuple[bool, str | None]:
    """Check source policy without changing quality decisions."""
    source_registered = bool(
        document.get("source_registered", True)
    )

    if not source_registered:
        return False, "source_unregistered"

    if sampling_scope == "experiment":
        return True, None

    if bool(
        document.get("release_eligible", False)
    ):
        return True, None

    if document.get("usage_policy") == "blocked":
        return False, "source_blocked"

    if document.get("license_status") in {
        "restricted",
        "unknown",
    }:
        return False, "license_restricted"

    return False, "license_not_verified"


def assign_sampling_weights(
    documents: list[dict],
    sampling_scope: str = "experiment",
) -> list[dict]:
    sampling_scope = normalize_sampling_scope(
        sampling_scope
    )

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

        policy_eligible, policy_reason = (
            assess_sampling_policy(
                document,
                sampling_scope,
            )
        )

        if base_weight <= 0:
            sampling_weight = 0.0
            sampling_status = "excluded"
            sampling_exclusion_reason = (
                "quality_rejected"
            )
        elif not policy_eligible:
            sampling_weight = 0.0
            sampling_status = "excluded"
            sampling_exclusion_reason = (
                policy_reason
            )
        else:
            balance_factor = 1 / math.sqrt(
                group_counts[group]
            )
            sampling_weight = round(
                base_weight * balance_factor,
                4,
            )
            sampling_status = (
                "eligible"
                if sampling_weight > 0
                else "excluded"
            )
            sampling_exclusion_reason = None

        balance_factor = 1 / math.sqrt(
            group_counts[group]
        )

        results.append(
            {
                **document,
                "sampling_scope": sampling_scope,
                "sampling_group": group,
                "base_sampling_weight": base_weight,
                "policy_eligible": policy_eligible,
                "policy_exclusion_reason": (
                    policy_reason
                ),
                "sampling_balance_factor": round(
                    balance_factor,
                    4,
                ),
                "sampling_weight": sampling_weight,
                "sampling_status": sampling_status,
                "sampling_exclusion_reason": (
                    sampling_exclusion_reason
                ),
            }
        )

    return results


def summarize_sampling(
    documents: list[dict],
    sampling_scope: str | None = None,
) -> dict:
    if sampling_scope is None:
        sampling_scope = next(
            (
                document.get(
                    "sampling_scope"
                )
                for document in documents
                if document.get("sampling_scope")
            ),
            "experiment",
        )

    sampling_scope = normalize_sampling_scope(
        sampling_scope
    )

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

    quality_excluded_count = sum(
        document.get(
            "base_sampling_weight",
            0.0,
        )
        <= 0
        for document in documents
    )
    policy_eligible_count = sum(
        document.get(
            "base_sampling_weight",
            0.0,
        )
        > 0
        and document.get(
            "policy_eligible",
            True,
        )
        for document in documents
    )
    policy_exclusion_counts = Counter(
        document.get(
            "policy_exclusion_reason",
            "unknown_policy_reason",
        )
        for document in documents
        if document.get(
            "base_sampling_weight",
            0.0,
        )
        > 0
        and not document.get(
            "policy_eligible",
            True,
        )
    )

    return {
        "sampling_scope": sampling_scope,
        "total": len(documents),
        "quality_excluded_count": (
            quality_excluded_count
        ),
        "policy_eligible_count": (
            policy_eligible_count
        ),
        "policy_excluded_count": sum(
            policy_exclusion_counts.values()
        ),
        "policy_exclusion_counts": dict(
            policy_exclusion_counts
        ),
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
