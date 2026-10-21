from math import ceil

from app.pipeline.rule_config import (
    load_quality_rules,
)
from app.pipeline.scoring import (
    content_signals,
)


def calibrated_score_document(
    document: dict,
    rules: dict | None = None,
) -> dict:
    if rules is None:
        rules = load_quality_rules()

    score = rules["base_score"]
    reasons = []

    length_rules = rules["length"]
    repetition_rules = rules["repetition"]
    removed_rules = rules["removed_text"]
    overlap_rules = rules["title_body_overlap"]
    link_rules = rules["link_ratio"]
    content_rules = rules.get(
        "content_density",
        {},
    )
    navigation_rules = rules.get(
        "navigation",
        {},
    )
    aggregation_rules = rules.get(
        "aggregation",
        {},
    )
    decision_rules = rules["decision"]

    signals = content_signals(
        document,
        rules=rules,
    )

    clean_text_chars = document.get(
        "clean_text_chars",
        0,
    )
    repeated_line_ratio = document.get(
        "repeated_line_ratio",
        0.0,
    )
    removed_text_ratio = document.get(
        "removed_text_ratio",
        0.0,
    )
    title_body_overlap = document.get(
        "title_body_overlap",
        0.0,
    )
    link_ratio = document.get(
        "link_ratio",
        0.0,
    )

    if clean_text_chars >= length_rules["long_min_chars"]:
        score += length_rules["long_bonus"]
        reasons.append(
            "正文长度达到100字符以上"
        )

    elif clean_text_chars >= length_rules[
        "normal_min_chars"
    ]:
        score += length_rules["normal_bonus"]
        reasons.append(
            "正文长度基本合格"
        )

    elif clean_text_chars >= length_rules[
        "short_min_chars"
    ]:
        score -= length_rules["short_penalty"]
        reasons.append("正文较短")

        short_rules = length_rules[
            "short_informative"
        ]

        if (
            title_body_overlap
            >= short_rules[
                "title_body_overlap_min"
            ]
            and repeated_line_ratio
            <= short_rules[
                "repeated_line_ratio_max"
            ]
            and removed_text_ratio
            < short_rules[
                "removed_text_ratio_max"
            ]
        ):
            score += length_rules[
                "short_informative_bonus"
            ]
            reasons.append(
                "短文本但结构完整"
            )

    else:
        score -= length_rules[
            "very_short_penalty"
        ]
        reasons.append("正文过短")

    if repeated_line_ratio > repetition_rules[
        "high_ratio"
    ]:
        score -= repetition_rules[
            "high_penalty"
        ]
        reasons.append("重复行比例过高")

    elif repeated_line_ratio > repetition_rules[
        "medium_ratio"
    ]:
        score -= repetition_rules[
            "medium_penalty"
        ]
        reasons.append("存在较多重复行")

    if removed_text_ratio > removed_rules[
        "high_ratio"
    ]:
        score -= removed_rules[
            "high_penalty"
        ]
        reasons.append(
            "清洗时删除了大量模板内容"
        )

    elif removed_text_ratio > removed_rules[
        "medium_ratio"
    ]:
        score -= removed_rules[
            "medium_penalty"
        ]
        reasons.append(
            "清洗时删除了一部分模板内容"
        )

    if title_body_overlap < overlap_rules[
        "low_ratio"
    ]:
        score -= overlap_rules[
            "low_penalty"
        ]
        reasons.append(
            "标题与正文相关性较低"
        )

    elif title_body_overlap < overlap_rules[
        "medium_ratio"
    ]:
        score -= overlap_rules[
            "medium_penalty"
        ]
        reasons.append(
            "标题与正文相关性一般"
        )

    if link_ratio > link_rules[
        "high_ratio"
    ]:
        if signals["content_rich"]:
            reasons.append(
                "正文内容密度较高，链接比例不降权"
            )
        else:
            score -= link_rules[
                "high_penalty"
            ]
            reasons.append(
                "链接内容比例过高"
            )

    if signals["content_rich"]:
        score += content_rules.get(
            "rich_bonus",
            15,
        )
        reasons.append("正文内容密度较高")

    if signals["navigation_like"]:
        score -= navigation_rules.get(
            "score_penalty",
            20,
        )
        reasons.append(
            "页面疑似目录、导航或索引"
        )

    if signals["aggregation_like"]:
        score -= aggregation_rules.get(
            "score_penalty",
            25,
        )
        reasons.append(
            "页面疑似博客或文章聚合页"
        )

    score = max(
        rules["score_bounds"]["min"],
        min(
            rules["score_bounds"]["max"],
            score,
        ),
    )

    if (
        signals["navigation_like"]
        or signals["aggregation_like"]
    ):
        content_decision = "reject"

    elif score >= decision_rules[
        "keep_min_score"
    ]:
        content_decision = "keep"

    elif score >= decision_rules[
        "review_min_score"
    ]:
        content_decision = "review"

    else:
        content_decision = "reject"

    if document.get(
        "is_near_duplicate",
        False,
    ):
        duplicate_status = "near_duplicate"

    elif document.get(
        "is_duplicate",
        False,
    ):
        duplicate_status = "exact_duplicate_audit"

    else:
        duplicate_status = "unique"

    if (
        duplicate_status == "near_duplicate"
        and content_decision != "reject"
    ):
        calibrated_decision = rules[
            "near_duplicate"
        ]["non_reject_decision"]

        policy_action = (
            "manual_duplicate_review"
        )

    else:
        calibrated_decision = content_decision
        policy_action = content_decision

    return {
        **document,
        "calibrated_quality_score": score,
        "content_decision": content_decision,
        "duplicate_status": duplicate_status,
        "calibrated_decision": calibrated_decision,
        "policy_action": policy_action,
        "calibrated_reasons": reasons,
        "content_rich": signals["content_rich"],
        "navigation_like": signals["navigation_like"],
        "aggregation_like": signals["aggregation_like"],
        "quality_rule_version": rules[
            "version"
        ],
    }


def calibrated_score_documents(
    documents: list[dict],
    rules: dict | None = None,
) -> list[dict]:
    """Score a batch and apply a bounded review budget.

    The per-document scorer keeps ambiguous cases as ``review``.  For a
    production batch, however, an unbounded review queue is not practical.
    The budget keeps the most ambiguous cases for audit and resolves overflow
    cases using a small, explicit score margin.  The original content score
    and reasons remain in every record for auditing.
    """

    if rules is None:
        rules = load_quality_rules()

    results = [
        calibrated_score_document(
            document,
            rules=rules,
        )
        for document in documents
    ]

    if not results:
        return results

    budget = rules.get(
        "review_budget",
        {},
    )
    max_rate = float(
        budget.get("max_rate", 1.0)
    )
    minimum_review_count = int(
        budget.get("minimum_review_count", 1)
    )
    max_review_count = max(
        minimum_review_count,
        ceil(len(results) * max_rate),
    )

    review_items = [
        item
        for item in results
        if item.get("calibrated_decision")
        == "review"
    ]

    if len(review_items) <= max_review_count:
        return results

    keep_min_score = rules["decision"][
        "keep_min_score"
    ]
    overflow_keep_margin = int(
        budget.get("overflow_keep_margin", 10)
    )
    midpoint = (
        rules["decision"]["keep_min_score"]
        + rules["decision"]["review_min_score"]
    ) / 2

    selected_ids = {
        item["id"]
        for item in sorted(
            review_items,
            key=lambda item: (
                abs(
                    item.get(
                        "calibrated_quality_score",
                        0,
                    )
                    - midpoint
                ),
                item.get(
                    "calibrated_quality_score",
                    0,
                ),
            ),
        )[:max_review_count]
    }

    for item in results:
        if (
            item.get("calibrated_decision")
            != "review"
            or item.get("id") in selected_ids
        ):
            continue

        score = item.get(
            "calibrated_quality_score",
            0,
        )
        if score >= keep_min_score - overflow_keep_margin:
            item["calibrated_decision"] = "keep"
            item["policy_action"] = (
                "review_budget_auto_keep"
            )
            item["calibrated_reasons"].append(
                "超出人工复核预算，按较高边界分自动保留"
            )
        else:
            item["calibrated_decision"] = "reject"
            item["policy_action"] = (
                "review_budget_auto_reject"
            )
            item["calibrated_reasons"].append(
                "超出人工复核预算，按边界分自动拒绝"
            )

    return results
