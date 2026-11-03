from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.download_failure import classify_error_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data/raw/html_urls_v3.jsonl"
)
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data/raw/html_benchmark_v3.jsonl"
)
DEFAULT_ERRORS_PATH = (
    PROJECT_ROOT / "data/raw/html_benchmark_v3_errors.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports/html_benchmark_v3/collection_quality_summary.json"
)


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
    path: str | Path,
) -> list[dict[str, Any]]:
    resolved_path = resolve_path(path)

    if not resolved_path.exists():
        return []

    return load_jsonl(resolved_path)


def _as_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")


def _duplicate_ids(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(
        _as_id(item)
        for item in rows
    )

    return sorted(
        item_id
        for item_id, count in counts.items()
        if item_id and count > 1
    )


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


def _sorted_counts(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counts = Counter(
        _group_value(item, field)
        for item in rows
    )

    return dict(sorted(counts.items()))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0

    return round(numerator / denominator, 4)


def _index_by_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}

    for item in rows:
        item_id = _as_id(item)

        if item_id in result:
            raise ValueError(
                f"发现重复 ID：{item_id}"
            )

        result[item_id] = item

    return result


def _coverage_by_field(
    manifest_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    manifest_map = {
        _as_id(item): item
        for item in manifest_rows
    }
    dataset_ids = {
        _as_id(item)
        for item in dataset_rows
    }
    error_ids = {
        _as_id(item)
        for item in error_rows
    }
    groups = set()

    for item in manifest_rows:
        groups.add(_group_value(item, field))

    for item in dataset_rows:
        groups.add(_group_value(item, field))

    coverage = {}

    for group in sorted(groups):
        expected_ids = {
            _as_id(item)
            for item in manifest_rows
            if _group_value(item, field) == group
        }
        downloaded_ids = {
            item_id
            for item_id in dataset_ids
            if (
                item_id in expected_ids
                and _group_value(
                    next(
                        (
                            item
                            for item in dataset_rows
                            if _as_id(item) == item_id
                        ),
                        manifest_map.get(item_id, {}),
                    ),
                    field,
                )
                == group
            )
        }
        failed_ids = {
            item_id
            for item_id in error_ids
            if (
                item_id in expected_ids
                and _group_value(
                    manifest_map.get(item_id, {}),
                    field,
                )
                == group
            )
        }

        expected_count = len(expected_ids)
        downloaded_count = len(downloaded_ids)
        failed_count = len(failed_ids)

        coverage[group] = {
            "expected": expected_count,
            "downloaded": downloaded_count,
            "errors": failed_count,
            "unaccounted": max(
                expected_count
                - downloaded_count
                - failed_count,
                0,
            ),
            "success_rate": _rate(
                downloaded_count,
                expected_count,
            ),
        }

    return coverage


def build_collection_quality_report(
    manifest_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]] | None = None,
    input_paths: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    error_rows = error_rows or []
    manifest_map = _index_by_id(manifest_rows)
    dataset_map = _index_by_id(dataset_rows)
    manifest_ids = set(manifest_map)
    dataset_ids = set(dataset_map)
    error_ids = {
        _as_id(item)
        for item in error_rows
        if _as_id(item)
    }

    classified_errors = []
    for error in error_rows:
        failure = classify_error_record(error)
        item_id = _as_id(error)
        manifest_item = manifest_map.get(item_id, {})
        classified_errors.append(
            {
                "id": item_id,
                "url": error.get(
                    "url",
                    manifest_item.get("url", ""),
                ),
                "error_type": str(
                    error.get(
                        "error_type",
                        "unknown",
                    )
                ),
                "error": str(error.get("error") or ""),
                **failure,
                "source": _group_value(
                    manifest_item,
                    "source",
                ),
                "language": _group_value(
                    manifest_item,
                    "language",
                ),
                "benchmark_category": _group_value(
                    manifest_item,
                    "benchmark_category",
                ),
            }
        )

    retryable_count = sum(
        bool(item["retryable"])
        for item in classified_errors
    )
    non_retryable_count = (
        len(classified_errors) - retryable_count
    )

    unaccounted_ids = sorted(
        manifest_ids - dataset_ids - error_ids
    )

    summary = {
        "manifest_count": len(manifest_rows),
        "downloaded_count": len(dataset_rows),
        "error_count": len(error_rows),
        "success_rate": _rate(
            len(dataset_ids & manifest_ids),
            len(manifest_ids),
        ),
        "error_rate": _rate(
            len(error_ids & manifest_ids),
            len(manifest_ids),
        ),
        "retryable_error_count": retryable_count,
        "non_retryable_error_count": non_retryable_count,
        "unaccounted_count": len(unaccounted_ids),
    }

    return {
        "report_version": "collection_quality_v1",
        "input": input_paths or {},
        "summary": summary,
        "distribution": {
            "manifest_by_language": _sorted_counts(
                manifest_rows,
                "language",
            ),
            "downloaded_by_language": _sorted_counts(
                dataset_rows,
                "language",
            ),
            "manifest_by_source": _sorted_counts(
                manifest_rows,
                "source",
            ),
            "downloaded_by_source": _sorted_counts(
                dataset_rows,
                "source",
            ),
            "manifest_by_category": _sorted_counts(
                manifest_rows,
                "benchmark_category",
            ),
            "downloaded_by_category": _sorted_counts(
                dataset_rows,
                "benchmark_category",
            ),
        },
        "coverage_by_language": _coverage_by_field(
            manifest_rows,
            dataset_rows,
            error_rows,
            "language",
        ),
        "coverage_by_source": _coverage_by_field(
            manifest_rows,
            dataset_rows,
            error_rows,
            "source",
        ),
        "coverage_by_category": _coverage_by_field(
            manifest_rows,
            dataset_rows,
            error_rows,
            "benchmark_category",
        ),
        "errors": classified_errors,
        "error_type_counts": dict(
            sorted(
                Counter(
                    item["error_type"]
                    for item in classified_errors
                ).items()
            )
        ),
        "error_class_counts": dict(
            sorted(
                Counter(
                    item["error_class"]
                    for item in classified_errors
                ).items()
            )
        ),
        "id_integrity": {
            "duplicate_manifest_ids": _duplicate_ids(
                manifest_rows
            ),
            "duplicate_dataset_ids": _duplicate_ids(
                dataset_rows
            ),
            "duplicate_error_ids": _duplicate_ids(
                error_rows
            ),
            "missing_manifest_ids": unaccounted_ids,
            "dataset_unknown_ids": sorted(
                dataset_ids - manifest_ids
            ),
            "error_unknown_ids": sorted(
                error_ids - manifest_ids
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 HTML 采集质量与下载失败报告"
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="URL 清单 JSONL 路径",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="成功下载数据 JSONL 路径",
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
    manifest_path = resolve_path(args.manifest)
    dataset_path = resolve_path(args.dataset)
    errors_path = resolve_path(args.errors)
    output_path = resolve_path(args.output)

    manifest_rows = load_jsonl(manifest_path)
    dataset_rows = load_jsonl(dataset_path)
    error_rows = load_optional_jsonl(errors_path)
    report = build_collection_quality_report(
        manifest_rows,
        dataset_rows,
        error_rows=error_rows,
        input_paths={
            "manifest": str(manifest_path),
            "dataset": str(dataset_path),
            "errors": str(errors_path),
        },
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
        )
        + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]
    print(
        f"采集质量报告完成：清单 {summary['manifest_count']} 条"
    )
    print(
        f"成功 {summary['downloaded_count']} 条，"
        f"失败 {summary['error_count']} 条"
    )
    print(
        f"成功率：{summary['success_rate']}，"
        f"可重试失败：{summary['retryable_error_count']} 条"
    )
    print(f"报告文件：{output_path}")


if __name__ == "__main__":
    main()
