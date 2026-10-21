PRODUCTION_DECISION_SOURCE = (
    "calibrated_scoring_v1"
)


def apply_production_decision(
    document: dict,
) -> dict:
    if "calibrated_decision" not in document:
        raise ValueError(
            f"缺少 calibrated_decision："
            f"{document.get('id')}"
        )

    return {
        **document,
        "production_decision": (
            document["calibrated_decision"]
        ),
        "production_quality_score": document.get(
            "calibrated_quality_score"
        ),
        "production_decision_source": (
            PRODUCTION_DECISION_SOURCE
        ),
    }