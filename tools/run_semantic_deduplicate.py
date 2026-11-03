"""Run exact, character n-gram, and embedding deduplication side by side."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from app.pipeline.deduplicate import annotate_exact_duplicates
from app.pipeline.near_deduplicate import annotate_near_duplicates
from app.pipeline.semantic_deduplicate import (
    DEFAULT_MODEL_NAME,
    annotate_semantic_duplicates,
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pair_key(pair: dict) -> tuple[str, str]:
    return tuple(sorted((pair["id_a"], pair["id_b"])))


def _pairs_from_exact(results: list[dict]) -> list[dict]:
    groups: dict[str, list[str]] = {}
    for item in results:
        group_id = item.get("duplicate_group")
        if group_id is not None:
            groups.setdefault(group_id, []).append(item["id"])

    pairs = []
    for ids in groups.values():
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                pairs.append(
                    {
                        "id_a": left,
                        "id_b": right,
                        "similarity": 1.0,
                    }
                )
    return pairs


def _method_summary(
    results: list[dict],
    pairs: list[dict],
    group_key: str,
) -> dict:
    groups = {
        item.get(group_key)
        for item in results
        if item.get(group_key) is not None
    }
    duplicate_key = {
        "duplicate_group": "is_duplicate",
        "near_duplicate_group": "is_near_duplicate",
        "semantic_duplicate_group": "is_semantic_duplicate",
    }[group_key]
    return {
        "candidate_pair_count": len(pairs),
        "duplicate_document_count": sum(
            bool(item.get(duplicate_key))
            for item in results
        ),
        "duplicate_group_count": len(groups),
    }


def _pair_set(pairs: list[dict]) -> set[tuple[str, str]]:
    return {_pair_key(pair) for pair in pairs}


def _pair_set_summary(
    name: str,
    pairs: set[tuple[str, str]],
) -> dict:
    return {
        "name": name,
        "candidate_pair_count": len(pairs),
        "pairs": [
            {"id_a": left, "id_b": right}
            for left, right in sorted(pairs)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行语义去重对照实验"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/html_benchmark_v5_baseline_v1/"
            "features.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/html_benchmark_v5_baseline_v1/"
            "semantic_dedup_experiment.json"
        ),
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path(
            "reports/html_benchmark_v5_baseline_v1/"
            "semantic_dedup_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--char-threshold",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    exact_results = annotate_exact_duplicates(documents)
    exact_pairs = _pairs_from_exact(exact_results)
    exact_seconds = round(time.perf_counter() - started, 4)

    started = time.perf_counter()
    char_results, char_pairs = annotate_near_duplicates(
        documents,
        threshold=args.char_threshold,
    )
    char_seconds = round(time.perf_counter() - started, 4)

    started = time.perf_counter()
    try:
        semantic_results, semantic_pairs = (
            annotate_semantic_duplicates(
                documents,
                threshold=args.embedding_threshold,
                model_name=args.model,
                batch_size=args.batch_size,
            )
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    semantic_seconds = round(time.perf_counter() - started, 4)

    exact_set = _pair_set(exact_pairs)
    char_set = _pair_set(char_pairs)
    semantic_set = _pair_set(semantic_pairs)
    intersection_set = char_set & semantic_set
    union_set = char_set | semantic_set

    report = {
        "experiment_version": "semantic_dedup_v1",
        "input": {
            "path": str(args.input),
            "document_count": len(documents),
        },
        "parameters": {
            "char_ngram_threshold": args.char_threshold,
            "embedding_threshold": args.embedding_threshold,
            "embedding_model": args.model,
            "batch_size": args.batch_size,
        },
        "methods": {
            "exact": {
                **_method_summary(
                    exact_results,
                    exact_pairs,
                    "duplicate_group",
                ),
                "elapsed_seconds": exact_seconds,
            },
            "char_ngram": {
                **_method_summary(
                    char_results,
                    char_pairs,
                    "near_duplicate_group",
                ),
                "elapsed_seconds": char_seconds,
            },
            "embedding": {
                **_method_summary(
                    semantic_results,
                    semantic_pairs,
                    "semantic_duplicate_group",
                ),
                "elapsed_seconds": semantic_seconds,
            },
        },
        "hybrid": {
            "intersection": _pair_set_summary(
                "char_ngram_and_embedding",
                intersection_set,
            ),
            "union": _pair_set_summary(
                "char_ngram_or_embedding",
                union_set,
            ),
        },
        "pair_reports": {
            "exact": exact_pairs,
            "char_ngram": char_pairs,
            "embedding": semantic_pairs,
        },
        "evaluation_note": (
            "当前报告是候选对与耗时对照；若要计算 precision/recall，"
            "还需要为文档对建立独立的 duplicate/non_duplicate 标签。"
        ),
    }

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pair_lines = []
    for method, pairs in report["pair_reports"].items():
        for pair in pairs:
            pair_lines.append(
                json.dumps(
                    {"method": method, **pair},
                    ensure_ascii=False,
                )
            )
    args.pairs_output.write_text(
        "\n".join(pair_lines) + ("\n" if pair_lines else ""),
        encoding="utf-8",
    )

    print(f"语义去重实验完成：{len(documents)} 条")
    print(
        "候选对数量："
        f"exact={len(exact_pairs)}, "
        f"char_ngram={len(char_pairs)}, "
        f"embedding={len(semantic_pairs)}"
    )
    print(f"实验报告：{args.output}")
    print(f"候选对明细：{args.pairs_output}")


if __name__ == "__main__":
    main()
