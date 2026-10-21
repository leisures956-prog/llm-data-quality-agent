from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.pipeline.calibrated_scoring import (
    calibrated_score_documents,
)
from app.pipeline.rule_config import (
    load_quality_rules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "config"
    / "benchmark_regression_policy.json"
)
RESULT_FILENAME = (
    "production_decisions_benchmark_v1.jsonl"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def load_json(path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_path(path)
    payload = json.loads(
        resolved_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"配置文件必须是 JSON 对象：{resolved_path}"
        )

    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = resolve_path(path)
    rows = []

    for line_number, line in enumerate(
        resolved_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        item = json.loads(line)

        if not isinstance(item, dict):
            raise ValueError(
                f"{resolved_path}:{line_number} 不是 JSON 对象"
            )

        if "id" not in item:
            raise ValueError(
                f"{resolved_path}:{line_number} 缺少 id 字段"
            )

        rows.append(item)

    return rows


def index_by_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed = {}

    for item in rows:
        item_id = str(item["id"])

        if item_id in indexed:
            raise ValueError(
                f"发现重复文档 ID：{item_id}"
            )

        indexed[item_id] = item

    return indexed


def get_decision(item: dict[str, Any]) -> str:
    return str(
        item.get("expected_decision")
        or item.get("production_decision")
        or item.get("calibrated_decision")
        or item.get("decision")
        or "unknown"
    )


def get_score(
    item: dict[str, Any],
) -> float | int | None:
    value = item.get(
        "expected_score",
        item.get(
            "calibrated_quality_score",
            item.get("quality_score"),
        ),
    )

    if value is None:
        return None

    number = float(value)

    if number.is_integer():
        return int(number)

    return number


def get_rule_version(
    item: dict[str, Any],
) -> str | None:
    value = item.get(
        "quality_rule_version"
    )

    if value is None:
        return None

    return str(value)


def decision_counts(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts = Counter(
        get_decision(item)
        for item in rows
    )

    return dict(sorted(counts.items()))


def average_score(
    rows: list[dict[str, Any]],
) -> float | None:
    scores = [
        float(score)
        for score in (
            get_score(item)
            for item in rows
        )
        if score is not None
    ]

    if not scores:
        return None

    return round(sum(scores) / len(scores), 4)


def score_feature_fixture(
    features_path: str | Path,
    rules_path: str | Path,
) -> list[dict[str, Any]]:
    features = load_jsonl(features_path)
    rules = load_quality_rules(
        str(resolve_path(rules_path))
    )

    return calibrated_score_documents(
        features,
        rules=rules,
    )


def _allowlisted_change(
    item_id: str,
    allowlist: dict[str, dict[str, Any]],
    change_type: str,
    change: dict[str, Any],
) -> bool:
    rule = allowlist.get(item_id)

    if rule is None:
        return False

    allowed_types = rule.get(
        "change_types",
        ["decision", "score"],
    )

    if change_type not in allowed_types:
        return False

    expected_after = rule.get(
        "after_decision"
    )

    if (
        change_type == "decision"
        and expected_after is not None
        and expected_after
        != change.get("after_decision")
    ):
        return False

    if change_type == "score":
        max_drop = rule.get(
            "max_score_drop"
        )

        if max_drop is not None:
            score_delta = change.get(
                "score_delta"
            )

            if (
                score_delta is not None
                and -score_delta > float(max_drop)
            ):
                return False

    return True


def evaluate_regression(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    policy: dict[str, Any],
    check_expected: bool = False,
) -> dict[str, Any]:
    old_map = index_by_id(old_rows)
    new_map = index_by_id(new_rows)

    old_ids = set(old_map)
    new_ids = set(new_map)
    added_ids = sorted(new_ids - old_ids)
    removed_ids = sorted(old_ids - new_ids)

    allowlist = {
        str(item["id"]): item
        for item in policy.get(
            "allowlisted_changes",
            [],
        )
        if isinstance(item, dict)
        and item.get("id") is not None
    }

    decision_changes = []
    score_changes = []
    missing_scores = []

    for item_id in sorted(old_ids & new_ids):
        before = old_map[item_id]
        after = new_map[item_id]

        before_decision = get_decision(before)
        after_decision = get_decision(after)
        before_score = get_score(before)
        after_score = get_score(after)

        if before_decision != after_decision:
            decision_changes.append(
                {
                    "id": item_id,
                    "before_decision": before_decision,
                    "after_decision": after_decision,
                    "before_score": before_score,
                    "after_score": after_score,
                    "allowlisted": (
                        _allowlisted_change(
                            item_id,
                            allowlist,
                            "decision",
                            {
                                "after_decision": (
                                    after_decision
                                )
                            },
                        )
                    ),
                }
            )

        if before_score != after_score:
            score_delta = None

            if (
                before_score is not None
                and after_score is not None
            ):
                score_delta = (
                    after_score - before_score
                )

            change = {
                "id": item_id,
                "before_score": before_score,
                "after_score": after_score,
                "score_delta": score_delta,
            }
            change["allowlisted"] = (
                _allowlisted_change(
                    item_id,
                    allowlist,
                    "score",
                    change,
                )
            )
            score_changes.append(change)

        if (
            before_score is not None
            and after_score is None
        ):
            missing_scores.append(item_id)

    old_counts = decision_counts(old_rows)
    new_counts = decision_counts(new_rows)
    old_average = average_score(old_rows)
    new_average = average_score(new_rows)

    allowlisted_score_ids = {
        change["id"]
        for change in score_changes
        if change["allowlisted"]
    }
    old_regression_rows = [
        item
        for item in old_rows
        if str(item["id"]) not in allowlisted_score_ids
    ]
    new_regression_rows = [
        item
        for item in new_rows
        if str(item["id"]) not in allowlisted_score_ids
    ]
    old_regression_average = average_score(
        old_regression_rows
    )
    new_regression_average = average_score(
        new_regression_rows
    )

    review_rate = round(
        new_counts.get("review", 0)
        / max(len(new_rows), 1),
        4,
    )

    violations = []

    if len(added_ids) > int(
        policy.get("max_added_ids", 0)
    ):
        violations.append(
            {
                "type": "added_ids",
                "value": added_ids,
            }
        )

    if len(removed_ids) > int(
        policy.get("max_removed_ids", 0)
    ):
        violations.append(
            {
                "type": "removed_ids",
                "value": removed_ids,
            }
        )

    unallowed_decision_changes = [
        change
        for change in decision_changes
        if not change["allowlisted"]
    ]

    max_decision_changes = int(
        policy.get(
            "max_decision_changes",
            0,
        )
    )

    if len(unallowed_decision_changes) > (
        max_decision_changes
    ):
        violations.append(
            {
                "type": "decision_changes",
                "value": unallowed_decision_changes,
            }
        )

    max_score_drop = float(
        policy.get(
            "max_score_drop_per_document",
            10,
        )
    )
    large_score_drops = [
        change
        for change in score_changes
        if not change["allowlisted"]
        and change["score_delta"] is not None
        and -change["score_delta"]
        > max_score_drop
    ]

    if large_score_drops:
        violations.append(
            {
                "type": "score_drop_per_document",
                "value": large_score_drops,
            }
        )

    if missing_scores:
        violations.append(
            {
                "type": "missing_scores",
                "value": missing_scores,
            }
        )

    if (
        old_regression_average is not None
        and new_regression_average is not None
        and old_regression_average
        - new_regression_average
        > float(
            policy.get(
                "max_average_score_drop",
                2,
            )
        )
    ):
        violations.append(
            {
                "type": "average_score_drop",
                "before": old_regression_average,
                "after": new_regression_average,
                "excluded_allowlisted_ids": sorted(
                    allowlisted_score_ids
                ),
            }
        )

    max_review_rate = float(
        policy.get(
            "max_review_rate",
            0.2,
        )
    )

    if review_rate > max_review_rate:
        violations.append(
            {
                "type": "review_rate",
                "value": review_rate,
                "max": max_review_rate,
            }
        )

    expected_rule_version = policy.get(
        "expected_rule_version"
    )
    actual_rule_versions = sorted(
        {
            version
            for version in (
                get_rule_version(item)
                for item in new_rows
            )
            if version
        }
    )

    if (
        expected_rule_version
        and actual_rule_versions
        != [str(expected_rule_version)]
    ):
        violations.append(
            {
                "type": "rule_version",
                "expected": str(
                    expected_rule_version
                ),
                "actual": actual_rule_versions,
            }
        )

    if check_expected:
        expected_total = policy.get(
            "expected_total"
        )

        if (
            expected_total is not None
            and len(new_rows)
            != int(expected_total)
        ):
            violations.append(
                {
                    "type": "expected_total",
                    "expected": int(expected_total),
                    "actual": len(new_rows),
                }
            )

        expected_counts = policy.get(
            "expected_decision_counts"
        )

        if (
            expected_counts is not None
            and new_counts
            != expected_counts
        ):
            violations.append(
                {
                    "type": "expected_decision_counts",
                    "expected": expected_counts,
                    "actual": new_counts,
                }
            )

    return {
        "status": (
            "passed"
            if not violations
            else "failed"
        ),
        "policy_version": policy.get(
            "version",
            "unknown",
        ),
        "old": {
            "total": len(old_rows),
            "decision_counts": old_counts,
            "average_score": old_average,
        },
        "new": {
            "total": len(new_rows),
            "decision_counts": new_counts,
            "average_score": new_average,
            "regression_average_score": {
                "before": old_regression_average,
                "after": new_regression_average,
                "excluded_allowlisted_ids": sorted(
                    allowlisted_score_ids
                ),
            },
            "review_rate": review_rate,
            "rule_versions": actual_rule_versions,
        },
        "added_ids": added_ids,
        "removed_ids": removed_ids,
        "decision_changes": decision_changes,
        "score_changes": score_changes,
        "violations": violations,
    }


def resolve_result_file(value: str | Path) -> Path:
    path = resolve_path(value)

    if path.is_dir():
        path = path / RESULT_FILENAME

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 Benchmark 结果文件：{path}"
        )

    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 Benchmark 自动化回归门禁"
    )
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        help="回归策略配置路径",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="轻量特征快照路径",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="基线结果 JSONL 路径",
    )
    parser.add_argument(
        "--old-run",
        default=None,
        help="完整 Benchmark 旧版本目录或 JSONL",
    )
    parser.add_argument(
        "--new-run",
        default=None,
        help="完整 Benchmark 新版本目录或 JSONL",
    )
    parser.add_argument(
        "--output",
        default="reports/benchmark_regression_gate.json",
        help="门禁报告输出路径",
    )

    args = parser.parse_args()
    policy = load_json(args.policy)

    full_run_mode = (
        args.old_run is not None
        or args.new_run is not None
    )

    if full_run_mode:
        if not args.old_run or not args.new_run:
            parser.error(
                "--old-run 和 --new-run 必须同时提供"
            )

        old_rows = load_jsonl(
            resolve_result_file(args.old_run)
        )
        new_rows = load_jsonl(
            resolve_result_file(args.new_run)
        )
        mode = "full_run"
        check_expected = False
    else:
        features_path = (
            args.features
            or policy["fixture_features_path"]
        )
        baseline_path = (
            args.baseline
            or policy["fixture_baseline_path"]
        )
        old_rows = load_jsonl(baseline_path)
        new_rows = score_feature_fixture(
            features_path,
            policy.get(
                "rules_path",
                "config/quality_rules.json",
            ),
        )
        mode = "fixture"
        check_expected = True

    report = evaluate_regression(
        old_rows,
        new_rows,
        policy,
        check_expected=check_expected,
    )
    report["mode"] = mode

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if report["status"] == "passed":
        print("Benchmark 回归门禁通过")
    else:
        print("Benchmark 回归门禁失败")

    print(f"模式：{mode}")
    print(f"样本数：{report['new']['total']}")
    print(
        "决策变化："
        f"{len(report['decision_changes'])} 条"
    )
    print(
        "新增 ID："
        f"{len(report['added_ids'])} 条"
    )
    print(
        "删除 ID："
        f"{len(report['removed_ids'])} 条"
    )
    print(
        "review 比例："
        f"{report['new']['review_rate']}"
    )
    print(f"报告文件：{output_path}")

    if report["violations"]:
        print("违规项：")
        for violation in report["violations"]:
            print(json.dumps(
                violation,
                ensure_ascii=False,
            ))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
