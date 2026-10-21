import re

try:
    from langdetect import detect as langdetect_detect
except ImportError:  # pragma: no cover - dependency is optional at import time
    langdetect_detect = None


LANGUAGE_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
}


TOPIC_KEYWORDS = {
    "technology": (
        "数据",
        "模型",
        "机器学习",
        "人工智能",
        "训练",
        "算法",
        "language model",
        "training",
        "data",
        "model",
    ),
    "medicine": (
        "医学",
        "医疗",
        "患者",
        "疾病",
        "治疗",
        "medicine",
        "patient",
        "clinical",
        "cancer",
    ),
    "finance": (
        "金融",
        "股票",
        "银行",
        "投资",
        "finance",
        "stock",
        "investment",
    ),
    "education": (
        "教育",
        "课程",
        "学习",
        "教学",
        "education",
        "course",
        "student",
    ),
}


KNOWN_DOMAINS = {
    "technology",
    "medicine",
    "finance",
    "education",
    "news",
    "navigation",
    "forum",
}


def normalize_language(language: str) -> str:
    normalized = (language or "unknown").strip().casefold()
    return LANGUAGE_ALIASES.get(normalized, normalized)


def detect_language(text: str) -> str:
    text = text or ""

    chinese_count = len(
        re.findall(r"[\u4e00-\u9fff]", text)
    )
    latin_count = len(
        re.findall(r"[A-Za-z]", text)
    )

    if chinese_count >= 2 and chinese_count >= latin_count:
        return "zh"

    if latin_count >= 5 and latin_count >= chinese_count * 2:
        return "en"

    if langdetect_detect is not None and text.strip():
        try:
            return normalize_language(langdetect_detect(text))
        except (ValueError, RuntimeError):
            pass

    return "unknown"


def infer_topic(document: dict) -> str:
    text = " ".join(
        [
            str(document.get("title", "")),
            str(document.get("clean_text", "")),
        ]
    ).casefold()

    topic_scores = {
        topic: sum(
            keyword.casefold() in text
            for keyword in keywords
        )
        for topic, keywords in TOPIC_KEYWORDS.items()
    }

    best_topic = max(
        topic_scores,
        key=topic_scores.get,
    )
    if topic_scores[best_topic] > 0:
        return best_topic

    domain = str(
        document.get("domain", "unknown")
    ).casefold()
    if domain in KNOWN_DOMAINS:
        return domain

    return "other"


def length_bucket(clean_text: str) -> str:
    character_count = len(clean_text or "")

    if character_count < 50:
        return "short"
    if character_count < 200:
        return "medium"
    return "long"


def quality_bucket(document: dict) -> str:
    decision = document.get(
        "production_decision"
    )

    if decision is None:
        decision = document.get(
            "calibrated_decision"
        )

    if decision is None:
        decision = document.get(
            "final_decision"
        )

    if decision is None:
        decision = document.get(
            "post_repair_decision",
            document.get("decision", "review"),
        )

    if decision in {
        "keep",
        "review",
        "reject",
    }:
        return decision

    return "review"

def classify_document(document: dict) -> dict:
    declared_language = normalize_language(
        str(document.get("language", "unknown"))
    )
    detected_language = detect_language(
        document.get("clean_text", "")
    )
    effective_language = (
        detected_language
        if detected_language != "unknown"
        else declared_language
    )

    return {
        **document,
        "declared_language": declared_language,
        "detected_language": detected_language,
        "language": effective_language,
        "topic": infer_topic(document),
        "length_bucket": length_bucket(
            document.get("clean_text", "")
        ),
        "quality_bucket": quality_bucket(document),
        "classification_method": "rules_and_language_detection",
        "decision_source": document.get(
            "production_decision_source",
            "legacy",
        ),
    }
