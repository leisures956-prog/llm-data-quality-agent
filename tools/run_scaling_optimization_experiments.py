"""Compare indexed near-dedup and parallel extraction experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from app.pipeline.near_deduplicate_indexed import (
    annotate_near_duplicates_indexed,
)
from app.pipeline.near_deduplicate import annotate_near_duplicates
from app.pipeline.parallel_extract import benchmark_extraction
from tools.build_scale_fixtures import load_jsonl, parse_sizes, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data/processed/performance_fixtures_v1/manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports/html_benchmark_v5_baseline_v1/"
    "scaling_optimization_v1.json"
)
DEFAULT_THRESHOLD = 0.75
DEFAULT_NEAR_MAX_DOCS = 2000
DEFAULT_WORKERS = (2, 4)


def _pair_set(pairs: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(pair["id_a"]), str(pair["id_b"]))
        for pair in pairs
    }


def _benchmark_near_dedup(
    documents: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    baseline_rows, baseline_pairs = annotate_near_duplicates(
        documents,
        threshold=threshold,
    )
    baseline_seconds = time.perf_counter() - started

    started = time.perf_counter()
    indexed_rows, indexed_pairs, indexed_stats = (
        annotate_near_duplicates_indexed(
            documents,
            threshold=threshold,
        )
    )
    indexed_seconds = time.perf_counter() - started

    baseline_pair_set = _pair_set(baseline_pairs)
    indexed_pair_set = _pair_set(indexed_pairs)
    theoretical_pairs = len(documents) * max(len(documents) - 1, 0) // 2
    intersection = len(baseline_pair_set & indexed_pair_set)
    recall = (
        intersection / len(baseline_pair_set)
        if baseline_pair_set
        else 1.0
    )
    return {
        "input_count": len(documents),
        "threshold": threshold,
        "baseline": {
            "elapsed_seconds": round(baseline_seconds, 4),
            "throughput_docs_per_second": round(
                len(documents) / baseline_seconds
                if baseline_seconds > 0
                else 0.0,
                4,
            ),
            "candidate_pair_count": theoretical_pairs,
            "scored_pair_count": theoretical_pairs,
            "duplicate_pair_count": len(baseline_pairs),
            "duplicate_document_count": sum(
                row.get("is_near_duplicate", False)
                for row in baseline_rows
            ),
        },
        "indexed": {
            "elapsed_seconds": round(indexed_seconds, 4),
            "throughput_docs_per_second": round(
                len(documents) / indexed_seconds
                if indexed_seconds > 0
                else 0.0,
                4,
            ),
            "duplicate_pair_count": len(indexed_pairs),
            "duplicate_document_count": sum(
                row.get("is_near_duplicate", False)
                for row in indexed_rows
            ),
            **indexed_stats,
        },
        "pair_parity": {
            "equal": baseline_pair_set == indexed_pair_set,
            "baseline_pair_count": len(baseline_pair_set),
            "indexed_pair_count": len(indexed_pair_set),
            "intersection_count": intersection,
            "recall_vs_baseline": round(recall, 4),
            "extra_pair_count": len(
                indexed_pair_set - baseline_pair_set
            ),
            "missing_pair_count": len(
                baseline_pair_set - indexed_pair_set
            ),
        },
    }


def run_experiments(
    manifest_path: str | Path,
    output_path: str | Path,
    sizes: tuple[int, ...] | None = None,
    worker_counts: tuple[int, ...] = DEFAULT_WORKERS,
    near_max_docs: int = DEFAULT_NEAR_MAX_DOCS,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    manifest_path = resolve_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("synthetic"):
        raise ValueError("优化实验只接受 synthetic=true 的样本清单")

    fixtures = manifest.get("fixtures", [])
    if sizes is not None:
        allowed = set(sizes)
        fixtures = [item for item in fixtures if item["size"] in allowed]
    if not fixtures:
        raise ValueError("没有匹配的性能样本")

    results = []
    for fixture in fixtures:
        documents = load_jsonl(fixture["path"])
        extraction = benchmark_extraction(
            documents,
            worker_counts=worker_counts,
        )
        serial_rows = extraction.pop("serial_rows")
        near_input = serial_rows[:near_max_docs]
        near_report = _benchmark_near_dedup(
            near_input,
            threshold=threshold,
        )
        near_report["capped"] = len(serial_rows) > near_max_docs
        near_report["cap"] = near_max_docs
        results.append(
            {
                "size": fixture["size"],
                "fixture_path": fixture["path"],
                "feature_extraction": extraction,
                "near_deduplication": near_report,
            }
        )

    report = {
        "optimization_version": "scaling_optimization_v1",
        "fixture_manifest": str(manifest_path),
        "fixture_mode": manifest.get("fixture_mode"),
        "synthetic": True,
        "near_dedup_threshold": threshold,
        "near_dedup_max_docs": near_max_docs,
        "worker_counts": list(worker_counts),
        "results": results,
        "ray_decision": {
            "status": "not_decided",
            "reason": (
                "先比较候选缩减和单机并行的收益；"
                "不因基线耗时直接引入 Ray"
            ),
        },
    }
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行近似去重候选缩减和正文抽取并行实验"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", default=None)
    parser.add_argument(
        "--workers",
        default="2,4",
        help="进程数列表，例如 2,4,8",
    )
    parser.add_argument(
        "--near-max-docs",
        type=int,
        default=DEFAULT_NEAR_MAX_DOCS,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    args = parser.parse_args()
    if args.near_max_docs <= 0:
        raise ValueError("--near-max-docs 必须是正整数")
    worker_counts = parse_sizes(args.workers)
    sizes = parse_sizes(args.sizes) if args.sizes else None

    report = run_experiments(
        args.manifest,
        args.output,
        sizes=sizes,
        worker_counts=worker_counts,
        near_max_docs=args.near_max_docs,
        threshold=args.threshold,
    )
    print(f"性能优化实验完成：{len(report['results'])} 个规模")
    for result in report["results"]:
        extraction = result["feature_extraction"]
        near = result["near_deduplication"]
        print(
            f"{result['size']} 条："
            f"串行抽取 {extraction['serial']['elapsed_seconds']} 秒；"
            f"近似去重候选缩减召回率 "
            f"{near['pair_parity']['recall_vs_baseline']}"
        )
    print(f"报告文件：{resolve_path(args.output)}")


if __name__ == "__main__":
    main()
