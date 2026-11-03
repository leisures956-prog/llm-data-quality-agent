"""Run the optional LLM-as-a-Judge pilot on boundary documents."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.pipeline.llm_judge import (
    HttpLLMJudgeClient,
    JudgeConfig,
    JsonlCache,
    LLMJudgeError,
    MockJudgeClient,
    judge_document,
    load_judge_config,
    select_boundary_documents,
    summarize_results,
)


DEFAULT_INPUT = Path(
    "data/processed/html_benchmark_v5_baseline_v1/"
    "production_decisions_benchmark_v1.jsonl"
)
DEFAULT_CONFIG = Path("config/llm_judge_v1.json")
DEFAULT_OUTPUT = Path("data/processed/llm_judge_results_v1.jsonl")
DEFAULT_CACHE = Path("data/processed/llm_judge_cache_v1.jsonl")
DEFAULT_SUMMARY = Path("reports/llm_judge_summary_v1.json")
DEFAULT_LABELS = Path("data/labels/html_benchmark_v5_audit_v1.jsonl")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _label_ids(
    labels_path: Path | None,
    split: str,
) -> set[str] | None:
    if labels_path is None:
        return None
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到标签文件：{labels_path}")
    labels = read_jsonl(labels_path)
    ids = {
        str(row["id"])
        for row in labels
        if row.get("split") == split
    }
    if not ids:
        raise ValueError(
            f"标签文件中没有 split={split} 的记录：{labels_path}"
        )
    return ids


def _filter_to_labels(
    documents: list[dict[str, Any]],
    labels_path: Path | None,
    split: str,
) -> list[dict[str, Any]]:
    ids = _label_ids(labels_path, split)
    if ids is None:
        return documents
    return [
        document
        for document in documents
        if str(document.get("id")) in ids
    ]


def _build_client(mode: str):
    if mode == "mock":
        return MockJudgeClient()
    if mode == "http":
        return HttpLLMJudgeClient()
    raise ValueError(f"不支持的 LLM Judge 模式：{mode}")


def run_judge(
    documents: list[dict[str, Any]],
    config: JudgeConfig,
    client: Any,
    cache: JsonlCache | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        machine_score = document.get(
            "calibrated_quality_score",
            document.get("quality_score"),
        )
        try:
            result = judge_document(
                document,
                config,
                client,
                cache=cache,
            )
        except LLMJudgeError as exc:
            result = {
                "document_id": str(document.get("id", "")),
                "status": "error",
                "judge_version": config.judge_version,
                "prompt_version": config.prompt_version,
                "model": config.model,
                "cached": False,
                "retry_count": config.max_retries,
                "latency_ms": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        rows.append(
            {
                "id": document.get("id"),
                "title": document.get("title", ""),
                "url": document.get("url", ""),
                "language": document.get("language", ""),
                "domain": document.get("domain", ""),
                "machine_decision": document.get(
                    "production_decision",
                    document.get(
                        "calibrated_decision",
                        document.get("decision"),
                    ),
                ),
                "machine_score": machine_score,
                "llm_judge": result,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行边界文档的 LLM-as-a-Judge 评估"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--mode",
        choices=("mock", "http"),
        default="mock",
    )
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--include-all", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_judge_config(args.config)
    if args.model:
        config = replace(config, model=args.model)

    documents = read_jsonl(args.input)
    documents = _filter_to_labels(
        documents,
        args.labels if str(args.labels) else None,
        args.split,
    )
    selected = select_boundary_documents(
        documents,
        config,
        include_all=args.include_all,
    )
    if args.max_docs > 0:
        selected = selected[: args.max_docs]

    cache = JsonlCache(args.cache)
    client = _build_client(args.mode)
    rows = run_judge(
        selected,
        config,
        client,
        cache=cache,
    )
    summary = summarize_results(
        [row["llm_judge"] for row in rows],
        input_path=str(args.input),
        mode=args.mode,
        config=config,
    )
    summary.update(
        {
            "selected_count": len(selected),
            "candidate_count_before_selection": len(documents),
            "labels_path": str(args.labels) if str(args.labels) else None,
            "split": args.split,
            "selection_mode": "all" if args.include_all else "boundary",
        }
    )

    write_jsonl(args.output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"LLM Judge 处理完成：{len(rows)} 条")
    print(f"模式：{args.mode}")
    print(f"状态统计：{summary['status_counts']}")
    print(f"决策统计：{summary['decision_counts']}")
    print(f"缓存命中：{summary['cache_hit_count']} 条")
    print(f"结果文件：{args.output}")
    print(f"统计文件：{args.summary_output}")


if __name__ == "__main__":
    main()
