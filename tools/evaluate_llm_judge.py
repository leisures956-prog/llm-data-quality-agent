"""Evaluate LLM-as-a-Judge results against reviewed human labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_DECISIONS = {"keep", "review", "reject"}
HUMAN_DECISIONS = {"keep", "reject"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSONL 解析失败：{path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL 每行必须是对象：{path}:{line_number}")
        rows.append(row)
    return rows


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _strict_keep_metrics(documents: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = sum(
        item["decision"] == "keep"
        and item["human_decision"] == "keep"
        for item in documents
    )
    false_positive = sum(
        item["decision"] == "keep"
        and item["human_decision"] == "reject"
        for item in documents
    )
    false_negative = sum(
        item["decision"] != "keep"
        and item["human_decision"] == "keep"
        for item in documents
    )
    true_negative = sum(
        item["decision"] != "keep"
        and item["human_decision"] == "reject"
        for item in documents
    )
    precision = _safe_rate(
        true_positive,
        true_positive + false_positive,
    )
    recall = _safe_rate(
        true_positive,
        true_positive + false_negative,
    )
    f1 = _safe_rate(
        2 * precision * recall,
        precision + recall,
    )
    return {
        "total": len(documents),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "accuracy": _safe_rate(
            true_positive + true_negative,
            len(documents),
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _confusion_matrix(documents: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        machine_decision: {
            human_decision: sum(
                item["decision"] == machine_decision
                and item["human_decision"] == human_decision
                for item in documents
            )
            for human_decision in ("keep", "reject")
        }
        for machine_decision in ("keep", "review", "reject")
    }


def evaluate_llm_judge(
    result_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    split: str = "dev",
) -> dict[str, Any]:
    result_by_id: dict[str, dict[str, Any]] = {}
    duplicate_result_ids: list[str] = []
    for row in result_rows:
        item_id = str(row.get("id", ""))
        if item_id in result_by_id:
            duplicate_result_ids.append(item_id)
        result_by_id[item_id] = row
    if duplicate_result_ids:
        raise ValueError(
            f"LLM Judge 结果中存在重复 ID：{sorted(set(duplicate_result_ids))}"
        )

    selected_labels = [
        row
        for row in label_rows
        if row.get("split") == split
        and row.get("reviewed") is True
    ]
    if not selected_labels:
        raise ValueError(f"人工标签中没有 split={split} 且 reviewed=true 的记录")

    evaluated: list[dict[str, Any]] = []
    uncovered_ids: list[str] = []
    error_cases: list[dict[str, Any]] = []
    for label in selected_labels:
        item_id = str(label.get("id", ""))
        row = result_by_id.get(item_id)
        if row is None:
            uncovered_ids.append(item_id)
            continue
        judge = row.get("llm_judge", {})
        if judge.get("status") != "ok":
            error_cases.append(
                {
                    "id": item_id,
                    "title": label.get("title", row.get("title", "")),
                    "status": judge.get("status", "unknown"),
                    "error_type": judge.get("error_type"),
                    "error_message": judge.get("error_message", ""),
                }
            )
            continue
        decision = judge.get("decision")
        if decision not in VALID_DECISIONS:
            error_cases.append(
                {
                    "id": item_id,
                    "title": label.get("title", row.get("title", "")),
                    "status": "invalid_decision",
                    "error_message": f"无效决策：{decision!r}",
                }
            )
            continue
        evaluated.append(
            {
                "id": item_id,
                "title": label.get("title", row.get("title", "")),
                "language": label.get("language", row.get("language", "unknown")),
                "topic": label.get(
                    "benchmark_category",
                    row.get("domain", "other"),
                ),
                "decision": decision,
                "human_decision": label["human_decision"],
                "confidence": judge.get("confidence"),
                "machine_score": row.get("machine_score"),
                "llm_quality_score": judge.get("quality_score"),
                "risk_flags": judge.get("risk_flags", []),
                "reasons": judge.get("reasons", []),
            }
        )

    for item in evaluated:
        if item["human_decision"] not in HUMAN_DECISIONS:
            raise ValueError(
                f"人工标签 {item['id']} 的 human_decision 无效："
                f"{item['human_decision']!r}"
            )

    human_keep_count = sum(
        item["human_decision"] == "keep"
        for item in evaluated
    )
    human_reject_count = sum(
        item["human_decision"] == "reject"
        for item in evaluated
    )
    hard_false_rejection_count = sum(
        item["decision"] == "reject"
        and item["human_decision"] == "keep"
        for item in evaluated
    )
    false_acceptance_count = sum(
        item["decision"] == "keep"
        and item["human_decision"] == "reject"
        for item in evaluated
    )
    review_count = sum(
        item["decision"] == "review"
        for item in evaluated
    )
    automatic_decision_count = sum(
        item["decision"] in {"keep", "reject"}
        for item in evaluated
    )

    return {
        "evaluation_version": "llm_judge_eval_v1",
        "split": split,
        "labeled_count": len(selected_labels),
        "evaluated_count": len(evaluated),
        "coverage": _safe_rate(
            len(evaluated),
            len(selected_labels),
        ),
        "uncovered_ids": uncovered_ids,
        "error_count": len(error_cases),
        "error_cases": error_cases,
        "machine_decision_counts": dict(
            Counter(item["decision"] for item in evaluated)
        ),
        "human_decision_counts": dict(
            Counter(item["human_decision"] for item in evaluated)
        ),
        "decision_cross_table": _confusion_matrix(evaluated),
        "strict_keep_metrics": _strict_keep_metrics(evaluated),
        "hard_false_rejection_count": hard_false_rejection_count,
        "hard_false_rejection_rate": _safe_rate(
            hard_false_rejection_count,
            human_keep_count,
        ),
        "false_acceptance_count": false_acceptance_count,
        "false_acceptance_rate": _safe_rate(
            false_acceptance_count,
            human_reject_count,
        ),
        "review_count": review_count,
        "review_rate": _safe_rate(review_count, len(evaluated)),
        "automatic_decision_count": automatic_decision_count,
        "automatic_decision_coverage": _safe_rate(
            automatic_decision_count,
            len(evaluated),
        ),
        "review_handling": {
            "strict_metrics_treat_review_as_not_keep": True,
            "hard_error_rates_count_only_direct_keep_or_reject": True,
        },
        "evaluated_cases": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估 LLM-as-a-Judge 与人工标签的一致性"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/processed/llm_judge_results_v1.jsonl"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/labels/html_benchmark_v5_audit_v1.jsonl"),
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/llm_judge_evaluation_v1.json"),
    )
    args = parser.parse_args()

    report = evaluate_llm_judge(
        read_jsonl(args.results),
        read_jsonl(args.labels),
        split=args.split,
    )
    report["source"] = {
        "results_file": str(args.results),
        "labels_file": str(args.labels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"LLM Judge 评估完成：{report['evaluated_count']} 条")
    print(f"覆盖率：{report['coverage']}")
    print(f"硬误杀率：{report['hard_false_rejection_rate']}")
    print(f"误保率：{report['false_acceptance_rate']}")
    print(f"人工复核：{report['review_count']} 条")
    print(f"报告文件：{args.output}")


if __name__ == "__main__":
    main()
