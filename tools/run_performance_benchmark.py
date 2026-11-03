"""Measure deterministic pipeline stages on synthetic scale fixtures."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from app.pipeline.calibrated_scoring import calibrated_score_document
from app.pipeline.classification import classify_document
from app.pipeline.deduplicate import (
    annotate_exact_duplicates,
    select_unique_documents,
)
from app.pipeline.extract import extract_document
from app.pipeline.features import compute_features
from app.pipeline.near_deduplicate import annotate_near_duplicates
from app.pipeline.rule_config import load_quality_rules
from app.pipeline.source_registry import load_source_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES_DIR = PROJECT_ROOT / "data/processed/performance_fixtures_v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports/html_benchmark_v5_baseline_v1/performance_v1.json"
)
DEFAULT_NEAR_DEDUP_MAX_DOCS = 2000


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = resolve_path(path)
    rows = []
    for line_number, line in enumerate(
        resolved_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(
                f"JSONL 每行必须是对象：{resolved_path}:{line_number}"
            )
        rows.append(item)
    return rows


def _measure(
    name: str,
    input_count: int,
    operation: Callable[[], Any],
    output_counter: Callable[[Any], int] | None = None,
    note: str | None = None,
) -> tuple[dict[str, Any], Any | None]:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:  # pragma: no cover - exercised by CLI failures
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return (
            {
                "stage": name,
                "status": "error",
                "input_count": input_count,
                "output_count": 0,
                "elapsed_seconds": round(elapsed, 4),
                "throughput_docs_per_second": 0.0,
                "peak_python_memory_mb": round(
                    peak / 1024 / 1024,
                    4,
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "note": note,
            },
            None,
        )

    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output_count = (
        output_counter(result)
        if output_counter is not None
        else (len(result) if hasattr(result, "__len__") else 0)
    )
    return (
        {
            "stage": name,
            "status": "ok",
            "input_count": input_count,
            "output_count": output_count,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_docs_per_second": round(
                input_count / elapsed
                if elapsed > 0
                else 0.0,
                4,
            ),
            "peak_python_memory_mb": round(
                peak / 1024 / 1024,
                4,
            ),
            "note": note,
        },
        result,
    )


def _build_features(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for document in documents:
        html = str(document.get("html") or "")
        extracted = extract_document(html)
        features = compute_features(
            html=html,
            title=extracted["title"],
            clean_text=extracted["clean_text"],
        )
        results.append(
            {
                **document,
                "title": extracted["title"],
                "clean_text": extracted["clean_text"],
                **features,
            }
        )
    return results


def _score_documents(
    documents: list[dict[str, Any]],
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        calibrated_score_document(document, rules=rules)
        for document in documents
    ]


def _exact_deduplicate(
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated = annotate_exact_duplicates(documents)
    return annotated, select_unique_documents(annotated)


def benchmark_fixture(
    fixture_path: str | Path,
    registry: dict[str, Any],
    rules: dict[str, Any],
    near_dedup_max_docs: int = DEFAULT_NEAR_DEDUP_MAX_DOCS,
) -> dict[str, Any]:
    fixture_path = resolve_path(fixture_path)
    load_started = time.perf_counter()
    documents = load_jsonl(fixture_path)
    load_elapsed = time.perf_counter() - load_started
    stages = [
        {
            "stage": "load",
            "status": "ok",
            "input_count": len(documents),
            "output_count": len(documents),
            "elapsed_seconds": round(load_elapsed, 4),
            "throughput_docs_per_second": round(
                len(documents) / load_elapsed
                if load_elapsed > 0
                else 0.0,
                4,
            ),
            "peak_python_memory_mb": None,
            "note": "JSONL 读取，不包含网络或大模型调用",
        }
    ]

    feature_stage, feature_rows = _measure(
        "feature_extraction",
        len(documents),
        lambda: _build_features(documents),
    )
    stages.append(feature_stage)
    if feature_rows is None:
        return {
            "fixture": str(fixture_path),
            "total": len(documents),
            "status": "error",
            "stages": stages,
        }

    exact_stage, exact_result = _measure(
        "exact_deduplicate",
        len(feature_rows),
        lambda: _exact_deduplicate(feature_rows),
        output_counter=lambda result: len(result[1]),
    )
    stages.append(exact_stage)
    if exact_result is None:
        return {
            "fixture": str(fixture_path),
            "total": len(documents),
            "status": "error",
            "stages": stages,
        }
    annotated_rows, unique_rows = exact_result

    near_input = unique_rows[:near_dedup_max_docs]
    near_note = None
    if len(unique_rows) > near_dedup_max_docs:
        near_note = (
            "近似去重为 O(n²)，仅测量前 "
            f"{near_dedup_max_docs} 条；未将 100k 全量强行展开"
        )
    near_stage, near_result = _measure(
        "near_deduplicate",
        len(near_input),
        lambda: annotate_near_duplicates(
            near_input,
            threshold=0.75,
        ),
        output_counter=lambda result: len(result[0]),
        note=near_note,
    )
    stages.append(near_stage)
    if near_result is None:
        return {
            "fixture": str(fixture_path),
            "total": len(documents),
            "status": "error",
            "stages": stages,
        }
    near_rows, _ = near_result

    score_stage, scored_rows = _measure(
        "calibrated_scoring",
        len(near_rows),
        lambda: _score_documents(near_rows, rules),
    )
    stages.append(score_stage)
    if scored_rows is None:
        return {
            "fixture": str(fixture_path),
            "total": len(documents),
            "status": "error",
            "stages": stages,
        }

    classification_stage, classified_rows = _measure(
        "classification",
        len(scored_rows),
        lambda: [classify_document(item) for item in scored_rows],
    )
    stages.append(classification_stage)
    if classified_rows is None:
        return {
            "fixture": str(fixture_path),
            "total": len(documents),
            "status": "error",
            "stages": stages,
        }

    decision_counts = Counter(
        item.get("content_decision", item.get("decision"))
        for item in classified_rows
        if item.get("content_decision", item.get("decision"))
    )
    duplicate_count = sum(
        item.get("is_duplicate", False)
        for item in annotated_rows
    )
    return {
        "fixture": str(fixture_path),
        "total": len(documents),
        "status": "ok",
        "decision_counts": dict(sorted(decision_counts.items())),
        "exact_duplicate_count": duplicate_count,
        "near_deduplicate_input_count": len(near_input),
        "near_deduplicate_capped": len(unique_rows) > near_dedup_max_docs,
        "stages": stages,
        "excluded_stages": [
            "llm_judge",
            "network_download",
        ],
    }


def run_performance_benchmark(
    manifest_path: str | Path,
    output_path: str | Path,
    near_dedup_max_docs: int = DEFAULT_NEAR_DEDUP_MAX_DOCS,
) -> dict[str, Any]:
    manifest_path = resolve_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("synthetic"):
        raise ValueError("性能基准只接受 synthetic=true 的样本清单")
    registry = load_source_registry(
        PROJECT_ROOT / "config/source_registry.json"
    )
    rules = load_quality_rules()
    results = [
        benchmark_fixture(
            fixture["path"],
            registry,
            rules,
            near_dedup_max_docs=near_dedup_max_docs,
        )
        for fixture in manifest.get("fixtures", [])
    ]
    report = {
        "performance_version": "performance_v1",
        "fixture_manifest": str(manifest_path),
        "synthetic": True,
        "fixture_mode": manifest.get("fixture_mode"),
        "quality_rule_version": rules.get("version"),
        "benchmark_scope": "deterministic_core_pipeline_only",
        "near_dedup_max_docs": near_dedup_max_docs,
        "results": results,
        "ray_decision": {
            "status": "not_decided",
            "reason": "先完成单机基准，再根据瓶颈决定是否引入 Ray",
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
        description="运行单机性能与规模基准，不包含 LLM 和网络调用"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_FIXTURES_DIR / "manifest.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--near-dedup-max-docs",
        type=int,
        default=DEFAULT_NEAR_DEDUP_MAX_DOCS,
    )
    args = parser.parse_args()
    if args.near_dedup_max_docs <= 0:
        raise ValueError("--near-dedup-max-docs 必须是正整数")

    report = run_performance_benchmark(
        args.manifest,
        args.output,
        near_dedup_max_docs=args.near_dedup_max_docs,
    )
    print(
        f"性能基准完成：{len(report['results'])} 个规模"
    )
    for result in report["results"]:
        print(
            f"{result['total']} 条：{result['status']}"
        )
        for stage in result["stages"]:
            print(
                f"  {stage['stage']}: "
                f"{stage['elapsed_seconds']} 秒，"
                f"{stage['throughput_docs_per_second']} docs/s"
            )
    print(f"报告文件：{resolve_path(args.output)}")


if __name__ == "__main__":
    main()
