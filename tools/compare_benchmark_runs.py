from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RESULT_FILENAME = "production_decisions_benchmark_v1.jsonl"


def resolve_result_file(value: str | Path) -> Path:
    path = Path(value)

    if path.is_dir():
        path = path / RESULT_FILENAME

    if not path.is_file():
        raise FileNotFoundError(f"找不到结果文件：{path}")

    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        item = json.loads(line)

        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} 不是 JSON 对象")

        if "id" not in item:
            raise ValueError(f"{path}:{line_number} 缺少 id 字段")

        rows.append(item)

    return rows


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}

    for item in rows:
        item_id = str(item["id"])

        if item_id in indexed:
            raise ValueError(f"发现重复文档 ID：{item_id}")

        indexed[item_id] = item

    return indexed


def get_decision(item: dict[str, Any]) -> str:
    return str(
        item.get("production_decision")
        or item.get("calibrated_decision")
        or item.get("decision")
        or "unknown"
    )


def get_score(item: dict[str, Any]) -> float | int | None:
    value = item.get(
        "calibrated_quality_score",
        item.get("quality_score"),
    )

    if value is None:
        return None

    number = float(value)

    if number.is_integer():
        return int(number)

    return number


def get_reasons(item: dict[str, Any]) -> list[str]:
    reasons = item.get(
        "calibrated_reasons",
        item.get("reasons", []),
    )

    if not isinstance(reasons, list):
        return []

    return [str(reason) for reason in reasons]


def get_rule_versions(rows: list[dict[str, Any]]) -> list[str]:
    versions = {
        str(item["quality_rule_version"])
        for item in rows
        if item.get("quality_rule_version")
    }

    return sorted(versions)


def decision_counts(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts = Counter(get_decision(item) for item in rows)
    return dict(sorted(counts.items()))


def compare_runs(
    old_run: str | Path,
    new_run: str | Path,
) -> dict[str, Any]:
    old_path = resolve_result_file(old_run)
    new_path = resolve_result_file(new_run)

    old_rows = load_jsonl(old_path)
    new_rows = load_jsonl(new_path)

    old_map = index_by_id(old_rows)
    new_map = index_by_id(new_rows)

    old_ids = set(old_map)
    new_ids = set(new_map)

    decision_changes = []
    score_changes = []
    reason_changes = []

    for item_id in sorted(old_ids & new_ids):
        before = old_map[item_id]
        after = new_map[item_id]

        before_decision = get_decision(before)
        after_decision = get_decision(after)

        before_score = get_score(before)
        after_score = get_score(after)

        before_reasons = set(get_reasons(before))
        after_reasons = set(get_reasons(after))

        if before_decision != after_decision:
            decision_changes.append(
                {
                    "id": item_id,
                    "title": after.get("title", ""),
                    "before_decision": before_decision,
                    "after_decision": after_decision,
                    "before_score": before_score,
                    "after_score": after_score,
                }
            )

        if before_score != after_score:
            score_delta = None

            if (
                before_score is not None
                and after_score is not None
            ):
                score_delta = after_score - before_score

            score_changes.append(
                {
                    "id": item_id,
                    "title": after.get("title", ""),
                    "before_score": before_score,
                    "after_score": after_score,
                    "score_delta": score_delta,
                }
            )

        added_reasons = sorted(after_reasons - before_reasons)
        removed_reasons = sorted(before_reasons - after_reasons)

        if added_reasons or removed_reasons:
            reason_changes.append(
                {
                    "id": item_id,
                    "title": after.get("title", ""),
                    "added_reasons": added_reasons,
                    "removed_reasons": removed_reasons,
                }
            )

    old_counts = decision_counts(old_rows)
    new_counts = decision_counts(new_rows)

    all_decisions = sorted(
        set(old_counts) | set(new_counts)
    )

    decision_delta = {
        decision: new_counts.get(decision, 0)
        - old_counts.get(decision, 0)
        for decision in all_decisions
    }

    new_reason_counts = Counter()

    for change in reason_changes:
        new_reason_counts.update(
            change["added_reasons"]
        )

    return {
        "old": {
            "result_file": str(old_path),
            "total": len(old_rows),
            "decision_counts": old_counts,
            "rule_versions": get_rule_versions(old_rows),
        },
        "new": {
            "result_file": str(new_path),
            "total": len(new_rows),
            "decision_counts": new_counts,
            "rule_versions": get_rule_versions(new_rows),
        },
        "decision_delta": decision_delta,
        "decision_changes": decision_changes,
        "score_changes": score_changes,
        "reason_changes": reason_changes,
        "new_reason_counts": dict(
            sorted(new_reason_counts.items())
        ),
        "added_ids": sorted(new_ids - old_ids),
        "removed_ids": sorted(old_ids - new_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较两个 Benchmark 版本的决策和评分变化"
    )

    parser.add_argument(
        "--old-run",
        required=True,
        help="旧版本结果目录或 JSONL 文件",
    )

    parser.add_argument(
        "--new-run",
        required=True,
        help="新版本结果目录或 JSONL 文件",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="对比报告输出路径",
    )

    args = parser.parse_args()

    report = compare_runs(
        args.old_run,
        args.new_run,
    )

    old_name = Path(args.old_run).name
    new_name = Path(args.new_run).name

    output_path = Path(
        args.output
        or f"reports/benchmark_compare_{old_name}_vs_{new_name}.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Benchmark 版本对比完成")
    print(f"决策变化：{len(report['decision_changes'])} 条")
    print(f"分数变化：{len(report['score_changes'])} 条")
    print(f"规则原因变化：{len(report['reason_changes'])} 条")
    print(f"报告文件：{output_path}")


if __name__ == "__main__":
    main()