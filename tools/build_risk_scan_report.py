"""Build an independent PII/source/license/contamination risk report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.pipeline.risk_scan import build_risk_report
from app.pipeline.source_registry import load_source_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/processed/html_benchmark_v5_baseline_v1/"
    "production_decisions_benchmark_v1.jsonl"
)
DEFAULT_REGISTRY = PROJECT_ROOT / "config/source_registry.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports/html_benchmark_v5_baseline_v1/risk_scan_v1.json"
)
DEFAULT_DETAILS_OUTPUT = (
    PROJECT_ROOT
    / "reports/html_benchmark_v5_baseline_v1/risk_scan_details_v1.jsonl"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{resolved_path}")

    rows = []
    for line_number, line in enumerate(
        resolved_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSONL 解析失败：{resolved_path}:{line_number}"
            ) from exc
        if not isinstance(item, dict):
            raise ValueError(
                f"JSONL 每行必须是对象：{resolved_path}:{line_number}"
            )
        if not str(item.get("id") or "").strip():
            raise ValueError(
                f"JSONL 缺少有效 id：{resolved_path}:{line_number}"
            )
        rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved_path = resolve_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成独立的数据风险扫描报告"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="可选的 Benchmark/测试集参考 JSONL，用于污染检测",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--details-output",
        type=Path,
        default=DEFAULT_DETAILS_OUTPUT,
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    registry_path = resolve_path(args.registry)
    reference_path = (
        resolve_path(args.reference)
        if args.reference is not None
        else None
    )
    documents = load_jsonl(input_path)
    reference_documents = (
        load_jsonl(reference_path)
        if reference_path is not None
        else []
    )
    registry = load_source_registry(registry_path)

    report = build_risk_report(
        documents,
        registry,
        reference_documents=reference_documents,
        input_paths={
            "input": str(input_path),
            "registry": str(registry_path),
        },
        reference_path=(
            str(reference_path) if reference_path is not None else None
        ),
    )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        args.details_output,
        report["documents"],
    )

    summary = report["summary"]
    print(f"风险扫描完成：{summary['total']} 条")
    print(
        "PII 风险："
        f"{summary['pii_risk_document_count']} 条；"
        "来源风险："
        f"{summary['source_risk_document_count']} 条；"
        "许可证风险："
        f"{summary['license_risk_document_count']} 条；"
        "污染风险："
        f"{summary['contamination_document_count']} 条"
    )
    print(f"汇总报告：{output_path}")
    print(
        "明细报告："
        f"{resolve_path(args.details_output)}"
    )


if __name__ == "__main__":
    main()
