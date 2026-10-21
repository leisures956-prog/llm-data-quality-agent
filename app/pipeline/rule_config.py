import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RULES_PATH = (
    PROJECT_ROOT / "config" / "quality_rules.json"
)


@lru_cache(maxsize=8)
def load_quality_rules(
    path: str | None = None,
) -> dict[str, Any]:
    rules_path = (
        Path(path)
        if path
        else DEFAULT_RULES_PATH
    )

    with rules_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        rules = json.load(file)

    validate_quality_rules(rules)

    return rules


def validate_quality_rules(
    rules: dict[str, Any],
) -> None:
    required_keys = {
        "version",
        "base_score",
        "score_bounds",
        "length",
        "repetition",
        "removed_text",
        "title_body_overlap",
        "link_ratio",
        "decision",
        "near_duplicate",
    }

    missing_keys = required_keys - set(rules)

    if missing_keys:
        raise ValueError(
            f"评分规则缺少字段：{sorted(missing_keys)}"
        )

    score_min = rules["score_bounds"]["min"]
    score_max = rules["score_bounds"]["max"]

    if score_min >= score_max:
        raise ValueError(
            "score_bounds 配置错误"
        )

    keep_min = rules["decision"]["keep_min_score"]
    review_min = rules["decision"]["review_min_score"]

    if keep_min <= review_min:
        raise ValueError(
            "keep_min_score 必须大于 review_min_score"
        )