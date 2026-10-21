from app.pipeline.classification import (
    classify_document,
    detect_language,
    infer_topic,
)


def test_detects_chinese_and_english():
    assert detect_language("这是一个数据质量分析页面。") == "zh"
    assert detect_language(
        "Data quality improves language model training."
    ) == "en"


def test_topic_keywords_are_used():
    document = {
        "title": "网页数据质量分析",
        "clean_text": "数据清洗可以提升模型训练质量。",
        "domain": "unknown",
    }

    assert infer_topic(document) == "technology"


def test_classification_falls_back_to_declared_language():
    document = {
        "id": "empty",
        "language": "zh-cn",
        "domain": "navigation",
        "title": "",
        "clean_text": "",
        "final_decision": "reject",
    }

    result = classify_document(document)

    assert result["detected_language"] == "unknown"
    assert result["language"] == "zh"
    assert result["topic"] == "navigation"
    assert result["quality_bucket"] == "reject"
