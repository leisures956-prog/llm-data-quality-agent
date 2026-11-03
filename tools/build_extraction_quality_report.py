from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURES_PATH = (
    PROJECT_ROOT / "data/processed/features.jsonl"
)
DEFAULT_DECISIONS_PATH = (
    PROJECT_ROOT
    / "data/processed/production_decisions_benchmark_v1.jsonl"
)
DEFAULT_ERRORS_PATH = (
    PROJECT_ROOT / "data/raw/html_benchmark_v2_errors.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/extraction_quality_summary.json"
)

SHORT_TEXT_CHARS = 100
HIGH_REMOVED_TEXT_RATIO = 0.7
HIGH_LINK_RATIO = 0.6
SINGLE_PARAGRAPH_MAX = 1


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


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


def load_optional_jsonl(
    path: str | Path | None,
) -> list[dict[str, Any]]:
    if path is None:
        return []

    resolved_path = resolve_path(path)

    if not resolved_path.exists():
        return []

    return load_jsonl(resolved_path)


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _decision(item: dict[str, Any]) -> str | None:
    value = (
        item.get("production_decision")
        or item.get("calibrated_decision")
        or item.get("decision")
    )

    if value is None:
        return None

    return str(value)


def _group_value(
    item: dict[str, Any],
    field: str,
) -> str:
    value = item.get(field)

    if field == "source" and not value:
        value = item.get("source_key")

    if value is None or not str(value).strip():
        return "unknown"

    return str(value)


def _annotate_feature(
    feature: dict[str, Any],
    decision: str | None,
) -> dict[str, Any]:
    clean_text_chars = _as_int(
        feature.get("clean_text_chars")
    )
    paragraph_count = _as_int(
        feature.get("paragraph_count")
    )
    sentence_count = _as_int(
        feature.get("sentence_count")
    )
    link_ratio = _as_float(
        feature.get("link_ratio")
    )
    removed_text_ratio = _as_float(
        feature.get("removed_text_ratio")
    )

    flags = []

    if clean_text_chars == 0:
        flags.append("empty_clean_text")
    elif clean_text_chars < SHORT_TEXT_CHARS:
        flags.append("short_clean_text")

    if removed_text_ratio >= HIGH_REMOVED_TEXT_RATIO:
        flags.append("high_removed_text_ratio")

    if link_ratio >= HIGH_LINK_RATIO:
        flags.append("high_link_ratio")

    if paragraph_count <= SINGLE_PARAGRAPH_MAX:
        flags.append("single_or_no_paragraph")

    return {
        "id": str(feature["id"]),
        "title": feature.get("title", ""),
        "source": _group_value(feature, "source"),
        "benchmark_category": _group_value(
            feature,
            "benchmark_category",
        ),
        "language": _group_value(feature, "language"),
        "decision": decision,
        "clean_text_chars": clean_text_chars,
        "paragraph_count": paragraph_count,
        "sentence_count": sentence_count,
        "link_ratio": round(link_ratio, 4),
        "removed_text_ratio": round(
            removed_text_ratio,
            4,
        ),
        "flags": flags,
    }


def _summarize_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_counts = Counter(
        row["decision"]
        for row in rows
        if row.get("decision")
    )

    return {
        "total": len(rows),
        "empty_clean_text_count": sum(
            "empty_clean_text" in row["flags"]
            for row in rows
        ),
        "short_clean_text_count": sum(
            "short_clean_text" in row["flags"]
            for row in rows
        ),
        "high_removed_text_ratio_count": sum(
            "high_removed_text_ratio" in row["flags"]
            for row in rows
        ),
        "high_link_ratio_count": sum(
            "high_link_ratio" in row["flags"]
            for row in rows
        ),
        "single_or_no_paragraph_count": sum(
            "single_or_no_paragraph" in row["flags"]
            for row in rows
        ),
        "flagged_document_count": sum(
            bool(row["flags"])
            for row in rows
        ),
        "decision_counts": dict(
            sorted(decision_counts.items())
        ),
    }


def _group_summary(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    groups = defaultdict(list)

    for row in rows:
        groups[row[field]].append(row)

    return {
        key: _summarize_rows(
            groups[key]
        )
        for key in sorted(groups)
    }


def build_extraction_quality_report(
    feature_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]] | None = None,
    error_rows: list[dict[str, Any]] | None = None,
    input_paths: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    decision_rows = decision_rows or []
    error_rows = error_rows or []

    decision_map = {}

    for item in decision_rows:
        item_id = str(item["id"])

        if item_id in decision_map:
            raise ValueError(
                f"发现重复决策 ID：{item_id}"
            )

        decision_map[item_id] = _decision(item)

    annotated_rows = []

    for feature in feature_rows:
        item_id = str(feature["id"])
        annotated_rows.append(
            _annotate_feature(
                feature,
                decision_map.get(item_id),
            )
        )

    feature_ids = {
        row["id"]
        for row in annotated_rows
    }
    decision_ids = set(decision_map)

    error_type_counts = Counter(
        str(item.get("error_type") or "unknown")
        for item in error_rows
    )

    flagged_rows = [
        {
            key: row[key]
            for key in (
                "id",
                "title",
                "source",
                "benchmark_category",
                "language",
                "decision",
                "clean_text_chars",
                "paragraph_count",
                "sentence_count",
                "link_ratio",
                "removed_text_ratio",
                "flags",
            )
        }
        for row in annotated_rows
        if row["flags"]
    ]

    return {
        "report_version": "extraction_quality_v1",
        "thresholds": {
            "short_text_chars": SHORT_TEXT_CHARS,
            "high_removed_text_ratio": (
                HIGH_REMOVED_TEXT_RATIO
            ),
            "high_link_ratio": HIGH_LINK_RATIO,
            "single_or_no_paragraph_max": (
                SINGLE_PARAGRAPH_MAX
            ),
        },
        "input": input_paths or {},
        "summary": _summarize_rows(annotated_rows),
        "download_errors": {
            "total": len(error_rows),
            "error_type_counts": dict(
                sorted(error_type_counts.items())
            ),
        },
        "decision_coverage": {
            "feature_count": len(feature_ids),
            "decision_count": len(decision_ids),
            "features_without_decision": sorted(
                feature_ids - decision_ids
            ),
            "decisions_without_features": sorted(
                decision_ids - feature_ids
            ),
        },
        "by_category": _group_summary(
            annotated_rows,
            "benchmark_category",
        ),
        "by_source": _group_summary(
            annotated_rows,
            "source",
        ),
        "by_language": _group_summary(
            annotated_rows,
            "language",
        ),
        "flagged_documents": flagged_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 HTML 正文抽取质量汇总报告"
    )
    parser.add_argument(
        "--features",
        default=str(DEFAULT_FEATURES_PATH),
        help="features.jsonl 路径",
    )
    parser.add_argument(
        "--decisions",
        default=str(DEFAULT_DECISIONS_PATH),
        help="正式决策 JSONL 路径，可选",
    )
    parser.add_argument(
        "--errors",
        default=str(DEFAULT_ERRORS_PATH),
        help="下载错误 JSONL 路径，可选",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="报告输出路径",
    )

    args = parser.parse_args()

    features_path = resolve_path(args.features)
    decisions_path = (
        resolve_path(args.decisions)
        if args.decisions
        else None
    )
    errors_path = (
        resolve_path(args.errors)
        if args.errors
        else None
    )

    feature_rows = load_jsonl(features_path)
    decision_rows = load_optional_jsonl(
        decisions_path
    )
    error_rows = load_optional_jsonl(errors_path)

    report = build_extraction_quality_report(
        feature_rows,
        decision_rows=decision_rows,
        error_rows=error_rows,
        input_paths={
            "features": str(features_path),
            "decisions": (
                str(decisions_path)
                if decisions_path
                else None
            ),
            "errors": (
                str(errors_path)
                if errors_path
                else None
            ),
        },
    )

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

    summary = report["summary"]
    print(
        f"抽取质量报告完成：{summary['total']} 条"
    )
    print(
        f"空正文：{summary['empty_clean_text_count']} 条"
    )
    print(
        f"正文过短：{summary['short_clean_text_count']} 条"
    )
    print(
        "高删除比例："
        f"{summary['high_removed_text_ratio_count']} 条"
    )
    print(
        "高链接比例："
        f"{summary['high_link_ratio_count']} 条"
    )
    print(
        f"下载错误：{len(error_rows)} 条"
    )
    print(f"报告文件：{output_path}")


if __name__ == "__main__":
    main()
