"""Evaluate one split of the frozen HTML Benchmark v5.

The evaluator runs one explicitly selected split at a time.  It joins
production decisions with the human labels, preserves ``review`` as a separate
machine decision, and reports direct hard errors separately from the strict
keep/reject metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.pipeline.evaluation import evaluate_documents


MACHINE_DECISIONS = {"keep", "review", "reject"}
HUMAN_DECISIONS = {"keep", "reject"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSONL 解析失败：{path}:{line_number}"
            ) from exc
    return rows


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _ensure_unique_ids(rows: list[dict], name: str) -> None:
    ids = [row.get("id") for row in rows]
    duplicate_ids = [
        item_id
        for item_id, count in Counter(ids).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"{name}中存在重复 ID：{duplicate_ids}")


def _select_and_validate_labels(
    labels: list[dict],
    split: str,
) -> list[dict]:
    selected = [
        label
        for label in labels
        if label.get("split") == split
    ]

    if not selected:
        raise ValueError(f"标签文件中没有 split={split} 的记录")

    _ensure_unique_ids(selected, f"split={split} 的人工标签")

    for label in selected:
        item_id = label.get("id")
        human_decision = label.get("human_decision")
        if human_decision not in HUMAN_DECISIONS:
            raise ValueError(
                f"人工标签 {item_id} 的 human_decision 无效："
                f"{human_decision!r}"
            )
        if label.get("reviewed") is not True:
            raise ValueError(
                f"人工标签 {item_id} 尚未标记 reviewed=true"
            )

        confidence = label.get("annotator_confidence")
        if not isinstance(confidence, (int, float)) or not (
            0 <= confidence <= 1
        ):
            raise ValueError(
                f"人工标签 {item_id} 的 annotator_confidence 无效："
                f"{confidence!r}"
            )

    return selected


def build_audit_documents(
    decision_rows: list[dict],
    label_rows: list[dict],
    split: str = "dev",
) -> list[dict]:
    """Join production decisions and labels for one evaluation split."""

    _ensure_unique_ids(decision_rows, "评分结果")
    selected_labels = _select_and_validate_labels(label_rows, split)
    decisions_by_id = {
        row["id"]: row
        for row in decision_rows
    }

    missing_ids = [
        label["id"]
        for label in selected_labels
        if label["id"] not in decisions_by_id
    ]
    if missing_ids:
        raise ValueError(
            f"评分结果中找不到这些标签 ID：{missing_ids}"
        )

    audit_documents = []
    for label in selected_labels:
        decision_row = decisions_by_id[label["id"]]
        machine_decision = decision_row.get(
            "production_decision",
            decision_row.get("decision"),
        )
        if machine_decision not in MACHINE_DECISIONS:
            raise ValueError(
                f"机器决策 {label['id']} 无效：{machine_decision!r}"
            )

        audit_documents.append(
            {
                **decision_row,
                "decision": machine_decision,
                "human_decision": label["human_decision"],
                "audit_split": split,
                "audit_error_type": label.get("error_type"),
                "audit_reason": label.get("review_reason", ""),
                "annotator_confidence": label.get(
                    "annotator_confidence"
                ),
            }
        )

    return audit_documents


def evaluate_html_benchmark_v5(
    decision_rows: list[dict],
    label_rows: list[dict],
    split: str = "dev",
) -> dict:
    documents = build_audit_documents(
        decision_rows,
        label_rows,
        split=split,
    )
    base_report = evaluate_documents(documents)

    human_keep_count = sum(
        item["human_decision"] == "keep"
        for item in documents
    )
    human_reject_count = sum(
        item["human_decision"] == "reject"
        for item in documents
    )
    hard_false_rejection_count = sum(
        item["decision"] == "reject"
        and item["human_decision"] == "keep"
        for item in documents
    )
    false_acceptance_count = sum(
        item["decision"] == "keep"
        and item["human_decision"] == "reject"
        for item in documents
    )
    review_count = sum(
        item["decision"] == "review"
        for item in documents
    )
    automatic_decision_count = sum(
        item["decision"] in {"keep", "reject"}
        for item in documents
    )

    error_cases = [
        {
            "id": item["id"],
            "title": item.get("title", ""),
            "machine_decision": item["decision"],
            "human_decision": item["human_decision"],
            "error_type": item.get("audit_error_type"),
            "reason": item.get("audit_reason", ""),
        }
        for item in documents
        if (
            item["decision"] == "reject"
            and item["human_decision"] == "keep"
        )
        or (
            item["decision"] == "keep"
            and item["human_decision"] == "reject"
        )
    ]

    review_cases = [
        {
            "id": item["id"],
            "title": item.get("title", ""),
            "human_decision": item["human_decision"],
            "error_type": item.get("audit_error_type"),
            "reason": item.get("audit_reason", ""),
        }
        for item in documents
        if item["decision"] == "review"
    ]

    bad_case_counts = Counter(
        item.get("audit_error_type") or "none"
        for item in documents
    )
    mismatch_bad_case_counts = Counter(
        item.get("audit_error_type") or "none"
        for item in documents
        if item["decision"] != item["human_decision"]
    )

    return {
        "evaluation_version": "html_benchmark_v5_eval_v1",
        "split": split,
        "total": len(documents),
        "machine_decision_counts": base_report[
            "machine_decision_counts"
        ],
        "human_decision_counts": base_report[
            "human_decision_counts"
        ],
        "decision_cross_table": base_report[
            "confusion_matrix"
        ],
        "strict_keep_metrics": base_report[
            "strict_keep_metrics"
        ],
        "review_handling": {
            "strict_metrics_treat_review_as_not_keep": True,
            "hard_error_rates_count_only_direct_keep_or_reject": True,
        },
        "hard_false_rejection_count": (
            hard_false_rejection_count
        ),
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
        "review_rate": _safe_rate(
            review_count,
            len(documents),
        ),
        "automatic_decision_count": automatic_decision_count,
        "automatic_decision_coverage": _safe_rate(
            automatic_decision_count,
            len(documents),
        ),
        "bad_case_counts": dict(bad_case_counts),
        "mismatch_bad_case_counts": dict(
            mismatch_bad_case_counts
        ),
        "error_cases": error_cases,
        "review_cases": review_cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 HTML Benchmark v5 的开发集或盲测集"
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(
            "data/processed/html_benchmark_v5_baseline_v1/"
            "production_decisions_benchmark_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "data/labels/html_benchmark_v5_audit_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/html_benchmark_v5_baseline_v1/"
            "dev_evaluation.json"
        ),
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "blind_test"],
        help="选择开发集或已完成标注的盲测集",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_html_benchmark_v5(
        read_jsonl(args.decisions),
        read_jsonl(args.labels),
        split=args.split,
    )
    report["audit_source"] = {
        "decisions_file": str(args.decisions),
        "labels_file": str(args.labels),
        "labeled_count": report["total"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"HTML Benchmark v5 评估完成：{report['total']} 条")
    print(f"机器决策：{report['machine_decision_counts']}")
    print(f"人工决策：{report['human_decision_counts']}")
    print(
        "硬误杀率："
        f"{report['hard_false_rejection_rate']} "
        f"({report['hard_false_rejection_count']} 条)"
    )
    print(
        "误保率："
        f"{report['false_acceptance_rate']} "
        f"({report['false_acceptance_count']} 条)"
    )
    print(
        "人工复核："
        f"{report['review_count']} 条，"
        f"比例 {report['review_rate']}"
    )
    print(
        "自动决策覆盖率："
        f"{report['automatic_decision_coverage']}"
    )
    print(f"报告文件：{args.output}")


if __name__ == "__main__":
    main()
