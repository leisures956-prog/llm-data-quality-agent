"""Evaluate exact, character, embedding, and hybrid pair deduplication."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.pipeline.deduplicate import normalize_text
from app.pipeline.semantic_deduplicate import (
    DEFAULT_MODEL_NAME,
    load_sentence_transformer,
)


VALID_LABELS = {"duplicate", "non_duplicate"}


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


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _validate_pairs(pair_rows: list[dict]) -> None:
    pair_ids = [row.get("pair_id") for row in pair_rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("语义去重对照集中存在重复 pair_id")
    if not pair_rows:
        raise ValueError("语义去重对照集为空")

    for row in pair_rows:
        pair_id = row.get("pair_id")
        if row.get("gold_label") not in VALID_LABELS:
            raise ValueError(
                f"{pair_id} 的 gold_label 无效："
                f"{row.get('gold_label')!r}"
            )
        if not str(row.get("text_a", "")).strip():
            raise ValueError(f"{pair_id} 的 text_a 为空")
        if not str(row.get("text_b", "")).strip():
            raise ValueError(f"{pair_id} 的 text_b 为空")


def _char_scores(pair_rows: list[dict]) -> list[float]:
    texts = []
    for row in pair_rows:
        texts.extend([row["text_a"], row["text_b"]])

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    return [
        float(
            cosine_similarity(
                matrix[index * 2],
                matrix[index * 2 + 1],
            )[0, 0]
        )
        for index in range(len(pair_rows))
    ]


def _encode_texts(texts: list[str], encoder: Any) -> np.ndarray:
    if not hasattr(encoder, "encode"):
        raise TypeError("encoder 必须提供 encode() 方法")
    try:
        embeddings = encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except TypeError:
        embeddings = encoder.encode(texts)
    return np.asarray(embeddings)


def _embedding_scores(
    pair_rows: list[dict],
    encoder: Any,
) -> list[float]:
    texts = []
    for row in pair_rows:
        texts.extend([row["text_a"], row["text_b"]])
    embeddings = _encode_texts(texts, encoder)
    return [
        float(
            cosine_similarity(
                embeddings[index * 2].reshape(1, -1),
                embeddings[index * 2 + 1].reshape(1, -1),
            )[0, 0]
        )
        for index in range(len(pair_rows))
    ]


def _binary_metrics(
    pair_rows: list[dict],
    predictions: list[bool],
) -> dict:
    gold = [
        row["gold_label"] == "duplicate"
        for row in pair_rows
    ]
    true_positive = sum(
        prediction and expected
        for prediction, expected in zip(predictions, gold)
    )
    false_positive = sum(
        prediction and not expected
        for prediction, expected in zip(predictions, gold)
    )
    false_negative = sum(
        not prediction and expected
        for prediction, expected in zip(predictions, gold)
    )
    true_negative = sum(
        not prediction and not expected
        for prediction, expected in zip(predictions, gold)
    )
    precision = _safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    recall = _safe_divide(
        true_positive,
        true_positive + false_negative,
    )
    f1 = _safe_divide(
        2 * precision * recall,
        precision + recall,
    )
    return {
        "total": len(pair_rows),
        "predicted_duplicate_count": sum(predictions),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "accuracy": _safe_divide(
            true_positive + true_negative,
            len(pair_rows),
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_pair_methods(
    pair_rows: list[dict],
    char_threshold: float = 0.75,
    embedding_threshold: float = 0.80,
    encoder: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    _validate_pairs(pair_rows)
    if not 0 <= char_threshold <= 1:
        raise ValueError("char_threshold 必须在 0 到 1 之间")
    if not 0 <= embedding_threshold <= 1:
        raise ValueError(
            "embedding_threshold 必须在 0 到 1 之间"
        )

    started = time.perf_counter()
    exact_predictions = [
        normalize_text(row["text_a"])
        == normalize_text(row["text_b"])
        for row in pair_rows
    ]
    exact_seconds = round(time.perf_counter() - started, 4)

    started = time.perf_counter()
    char_values = _char_scores(pair_rows)
    char_predictions = [
        score >= char_threshold
        for score in char_values
    ]
    char_seconds = round(time.perf_counter() - started, 4)

    if encoder is None:
        encoder = load_sentence_transformer(model_name)
    started = time.perf_counter()
    embedding_values = _embedding_scores(pair_rows, encoder)
    embedding_predictions = [
        score >= embedding_threshold
        for score in embedding_values
    ]
    embedding_seconds = round(time.perf_counter() - started, 4)

    predictions = {
        "exact": exact_predictions,
        "char_ngram": char_predictions,
        "embedding": embedding_predictions,
        "hybrid_intersection": [
            char_prediction and embedding_prediction
            for char_prediction, embedding_prediction
            in zip(char_predictions, embedding_predictions)
        ],
        "hybrid_union": [
            char_prediction or embedding_prediction
            for char_prediction, embedding_prediction
            in zip(char_predictions, embedding_predictions)
        ],
    }
    metrics = {
        method: _binary_metrics(pair_rows, values)
        for method, values in predictions.items()
    }
    metrics["exact"]["elapsed_seconds"] = exact_seconds
    metrics["char_ngram"]["elapsed_seconds"] = char_seconds
    metrics["embedding"]["elapsed_seconds"] = embedding_seconds
    metrics["hybrid_intersection"]["elapsed_seconds"] = round(
        char_seconds + embedding_seconds,
        4,
    )
    metrics["hybrid_union"]["elapsed_seconds"] = round(
        char_seconds + embedding_seconds,
        4,
    )

    details = []
    for index, row in enumerate(pair_rows):
        details.append(
            {
                "pair_id": row["pair_id"],
                "pair_type": row.get("pair_type"),
                "gold_label": row["gold_label"],
                "char_similarity": round(char_values[index], 4),
                "embedding_similarity": round(
                    embedding_values[index],
                    4,
                ),
                "predictions": {
                    method: values[index]
                    for method, values in predictions.items()
                },
            }
        )

    return {
        "evaluation_version": "semantic_pair_eval_v1",
        "pair_count": len(pair_rows),
        "pair_type_counts": dict(
            Counter(row.get("pair_type", "unknown") for row in pair_rows)
        ),
        "label_counts": dict(
            Counter(row["gold_label"] for row in pair_rows)
        ),
        "thresholds": {
            "char_ngram": char_threshold,
            "embedding": embedding_threshold,
        },
        "embedding_model": model_name,
        "metrics": metrics,
        "details": details,
        "label_source_note": (
            "当前对照集为构造式实验样本，用于验证方法链路；"
            "不能替代真实网页对的独立人工标注。"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估语义去重方法"
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/labels/semantic_pairs_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/html_benchmark_v5_baseline_v1/"
            "semantic_pair_evaluation.json"
        ),
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=Path(
            "reports/html_benchmark_v5_baseline_v1/"
            "semantic_pair_evaluation_details.jsonl"
        ),
    )
    parser.add_argument("--char-threshold", type=float, default=0.75)
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_rows = read_jsonl(args.pairs)
    try:
        report = evaluate_pair_methods(
            pair_rows,
            char_threshold=args.char_threshold,
            embedding_threshold=args.embedding_threshold,
            model_name=args.model,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    report["input"] = {
        "pairs_file": str(args.pairs),
        "pair_count": len(pair_rows),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.details_output.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in report["details"]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"语义去重对照评估完成：{report['pair_count']} 对")
    for method, metrics in report["metrics"].items():
        print(
            f"{method}: "
            f"precision={metrics['precision']}, "
            f"recall={metrics['recall']}, "
            f"f1={metrics['f1']}"
        )
    print(f"汇总报告：{args.output}")
    print(f"明细报告：{args.details_output}")


if __name__ == "__main__":
    main()
