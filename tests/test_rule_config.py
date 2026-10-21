import copy

from app.pipeline.calibrated_scoring import (
    calibrated_score_document,
)
from app.pipeline.rule_config import (
    load_quality_rules,
)


def test_quality_rules_can_be_loaded():
    rules = load_quality_rules()

    assert rules["version"] == (
        "quality_rules_v3_content_density_aggregation"
    )
    assert rules["decision"][
        "keep_min_score"
    ] == 70


def test_custom_threshold_changes_decision():
    rules = copy.deepcopy(
        load_quality_rules()
    )

    rules["decision"][
        "keep_min_score"
    ] = 80

    document = {
        "id": "config_test",
        "clean_text_chars": 100,
        "repeated_line_ratio": 0.0,
        "removed_text_ratio": 0.2,
        "title_body_overlap": 1.0,
        "link_ratio": 0.0,
        "is_duplicate": False,
        "is_near_duplicate": False,
    }

    result = calibrated_score_document(
        document,
        rules=rules,
    )

    assert result[
        "calibrated_quality_score"
    ] == 75

    assert result[
        "calibrated_decision"
    ] == "review"
