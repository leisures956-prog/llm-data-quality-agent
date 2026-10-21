DEFAULT_CONTENT_POLICY = {
    "rich_min_chars": 3000,
    "rich_min_paragraphs": 15,
    "rich_min_sentences": 30,
    "rich_bonus": 15,
}

DEFAULT_NAVIGATION_POLICY = {
    "high_link_ratio": 0.80,
    "high_link_max_paragraphs": 12,
    "medium_link_ratio": 0.60,
    "medium_link_max_paragraphs": 10,
    "medium_link_max_sentences": 20,
    "thin_max_chars": 1200,
    "thin_min_paragraphs": 5,
    "thin_max_sentences": 20,
    "thin_min_link_ratio": 0.25,
    "small_index_max_chars": 1800,
    "small_index_min_paragraphs": 4,
    "small_index_max_sentences": 12,
    "small_index_min_link_ratio": 0.40,
    "score_penalty": 20,
}

DEFAULT_AGGREGATION_POLICY = {
    "title_terms": (
        "blog",
        "blogs",
        "insider",
        "latest news",
        "news",
    ),
    "max_title_words": 5,
    "min_paragraphs": 10,
    "max_average_paragraph_chars": 250,
    "min_link_ratio": 0.20,
    "score_penalty": 25,
}


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def content_signals(
    document: dict,
    rules: dict | None = None,
) -> dict[str, bool]:
    """Return content-density and navigation signals for one document.

    Link-heavy documentation pages can still contain valuable continuous
    prose.  Navigation pages, on the other hand, tend to have very high link
    ratios together with few paragraphs or sentences.  Keeping these signals
    explicit lets the calibrated scorer explain why it made a decision.
    """

    rules = rules or {}
    content_rules = {
        **DEFAULT_CONTENT_POLICY,
        **rules.get("content_density", {}),
    }
    navigation_rules = {
        **DEFAULT_NAVIGATION_POLICY,
        **rules.get("navigation", {}),
    }
    aggregation_rules = {
        **DEFAULT_AGGREGATION_POLICY,
        **rules.get("aggregation", {}),
    }

    clean_text_chars = _as_int(
        document.get("clean_text_chars")
    )
    paragraph_count = _as_int(
        document.get("paragraph_count")
    )
    sentence_count = _as_int(
        document.get("sentence_count")
    )
    link_ratio = _as_float(
        document.get("link_ratio")
    )
    title = str(
        document.get("title", "")
    ).strip().casefold()
    title_words = [
        word
        for word in title.split()
        if word
    ]
    average_paragraph_chars = (
        clean_text_chars
        / max(paragraph_count, 1)
    )

    content_rich = (
        clean_text_chars
        >= content_rules["rich_min_chars"]
        and paragraph_count
        >= content_rules["rich_min_paragraphs"]
        and sentence_count
        >= content_rules["rich_min_sentences"]
    )

    high_link_navigation = (
        link_ratio
        >= navigation_rules["high_link_ratio"]
        and paragraph_count
        <= navigation_rules["high_link_max_paragraphs"]
    )
    medium_link_navigation = (
        link_ratio
        >= navigation_rules["medium_link_ratio"]
        and paragraph_count
        <= navigation_rules["medium_link_max_paragraphs"]
        and sentence_count
        <= navigation_rules["medium_link_max_sentences"]
    )
    thin_navigation = (
        clean_text_chars
        <= navigation_rules["thin_max_chars"]
        and paragraph_count
        >= navigation_rules["thin_min_paragraphs"]
        and sentence_count
        <= navigation_rules["thin_max_sentences"]
        and link_ratio
        >= navigation_rules["thin_min_link_ratio"]
    )
    small_index_navigation = (
        clean_text_chars
        <= navigation_rules["small_index_max_chars"]
        and paragraph_count
        >= navigation_rules["small_index_min_paragraphs"]
        and sentence_count
        <= navigation_rules["small_index_max_sentences"]
        and link_ratio
        >= navigation_rules["small_index_min_link_ratio"]
    )
    aggregation_like = (
        any(
            term in title
            for term in aggregation_rules[
                "title_terms"
            ]
        )
        and len(title_words)
        <= aggregation_rules[
            "max_title_words"
        ]
        and paragraph_count
        >= aggregation_rules[
            "min_paragraphs"
        ]
        and average_paragraph_chars
        <= aggregation_rules[
            "max_average_paragraph_chars"
        ]
        and link_ratio
        >= aggregation_rules[
            "min_link_ratio"
        ]
    )

    return {
        "content_rich": content_rich,
        "navigation_like": (
            not content_rich
            and (
                high_link_navigation
                or medium_link_navigation
                or thin_navigation
                or small_index_navigation
            )
        ),
        "aggregation_like": aggregation_like,
    }


def score_document(document: dict) -> dict:
    score = 50
    reasons = []

    signals = content_signals(document)

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

    is_near_duplicate = document.get(
        "is_near_duplicate",
        False,
    )

    if clean_text_chars >= 100:
        score += 25
        reasons.append("正文长度达到100字符以上")
    elif clean_text_chars >= 50:
        score += 10
        reasons.append("正文长度基本合格")
    else:
        score -= 20
        reasons.append("正文过短")

    if repeated_line_ratio > 0.5:
        score -= 30
        reasons.append("重复行比例过高")
    elif repeated_line_ratio > 0.2:
        score -= 15
        reasons.append("存在较多重复行")

    if removed_text_ratio > 0.7:
        score -= 25
        reasons.append("清洗时删除了大量模板内容")
    elif removed_text_ratio > 0.5:
        score -= 10
        reasons.append("清洗时删除了一部分模板内容")

    if title_body_overlap < 0.2:
        score -= 15
        reasons.append("标题与正文相关性较低")
    elif title_body_overlap < 0.5:
        score -= 5
        reasons.append("标题与正文相关性一般")

    if link_ratio > 0.4:
        if signals["content_rich"]:
            reasons.append(
                "正文内容密度较高，链接比例不降权"
            )
        else:
            score -= 15
            reasons.append("链接内容比例过高")

    if signals["content_rich"]:
        score += DEFAULT_CONTENT_POLICY[
            "rich_bonus"
        ]
        reasons.append("正文内容密度较高")

    if signals["navigation_like"]:
        score -= DEFAULT_NAVIGATION_POLICY[
            "score_penalty"
        ]
        reasons.append("页面疑似目录、导航或索引")

    if signals["aggregation_like"]:
        score -= DEFAULT_AGGREGATION_POLICY[
            "score_penalty"
        ]
        reasons.append("页面疑似博客或文章聚合页")

    if is_near_duplicate:
        score -= 10
        reasons.append("与其他文档内容高度相似，需要复核")

    score = max(0, min(100, score))

    if (
        signals["navigation_like"]
        or signals["aggregation_like"]
    ):
        decision = "reject"
    elif score >= 70:
        decision = "keep"
    elif score >= 40:
        decision = "review"
    else:
        decision = "reject"

    return {
        **document,
        "quality_score": score,
        "decision": decision,
        "reasons": reasons,
        "content_rich": signals["content_rich"],
        "navigation_like": signals["navigation_like"],
        "aggregation_like": signals["aggregation_like"],
    }


def score_documents(
    documents: list[dict],
) -> list[dict]:
    return [
        score_document(document)
        for document in documents
    ]
