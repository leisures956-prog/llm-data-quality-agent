from app.pipeline.classification import (
    classify_document,
)
from app.pipeline.decision_policy import (
    apply_production_decision,
)


def test_production_decision_is_added():
    document = {
        "id": "doc",
        "decision": "reject",
        "calibrated_decision": "review",
        "calibrated_quality_score": 40,
        "final_decision": "keep",
    }

    result = apply_production_decision(document)

    assert result["production_decision"] == "review"
    assert result["final_decision"] == "keep"
    assert (
        result["production_decision_source"]
        == "calibrated_scoring_v1"
    )


def test_classification_uses_production_decision():
    document = {
        "id": "doc",
        "language": "zh",
        "title": "数据质量",
        "clean_text": "数据质量分析。",
        "decision": "reject",
        "final_decision": "keep",
        "production_decision": "review",
        "production_decision_source": (
            "calibrated_scoring_v1"
        ),
    }

    result = classify_document(document)

    assert result["quality_bucket"] == "review"
    assert (
        result["decision_source"]
        == "calibrated_scoring_v1"
    )