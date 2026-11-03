"""Build a small, reviewable real-web pair set for semantic deduplication.

The generated JSONL is an annotation template.  It deliberately leaves
``gold_label`` and ``label_reason`` empty so that a human can label each pair
without changing the production benchmark or the synthetic control set.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_INPUT = Path(
    "data/processed/html_benchmark_v5_baseline_v1/features.jsonl"
)
DEFAULT_OUTPUT = Path("data/labels/semantic_pairs_real_v1.jsonl")
DEFAULT_PAIR_COUNT = 20


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到特征文件：{path}")

    rows: list[dict[str, Any]] = []
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


def _pair_key(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((str(left["id"]), str(right["id"]))))


def _rank_pairs(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = [str(row.get("clean_text", "")) for row in documents]
    if len(documents) < 2:
        return []

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(matrix)

    pairs: list[dict[str, Any]] = []
    for left_index, right_index in itertools.combinations(
        range(len(documents)),
        2,
    ):
        left = documents[left_index]
        right = documents[right_index]
        pairs.append(
            {
                "left_index": left_index,
                "right_index": right_index,
                "pair_key": _pair_key(left, right),
                "char_similarity": round(
                    float(similarity_matrix[left_index, right_index]),
                    4,
                ),
                "same_domain": left.get("domain") == right.get("domain"),
                "same_language": left.get("language") == right.get("language"),
            }
        )

    return pairs


def _select_pairs(
    documents: list[dict[str, Any]],
    ranked_pairs: list[dict[str, Any]],
    pair_count: int,
) -> list[dict[str, Any]]:
    if pair_count <= 0:
        raise ValueError("pair_count 必须大于 0")

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()

    def add_candidates(
        candidates: list[dict[str, Any]],
        limit: int | None = None,
    ) -> None:
        added = 0
        for candidate in candidates:
            if len(selected) >= pair_count:
                return
            if limit is not None and added >= limit:
                return
            if candidate["pair_key"] in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(candidate["pair_key"])
            added += 1

    # First keep the most similar pairs: these are the highest-value candidates
    # for finding paraphrases or partial-overlap duplicates.
    add_candidates(
        sorted(
            ranked_pairs,
            key=lambda item: item["char_similarity"],
            reverse=True,
        )[:8]
    )

    # Add same-domain pairs that may be topically related but not duplicates.
    same_domain = [
        item
        for item in ranked_pairs
        if item["same_domain"]
    ]
    add_candidates(
        sorted(
            same_domain,
            key=lambda item: item["char_similarity"],
            reverse=True,
        ),
        limit=6,
    )

    # Add low-similarity cross-domain negatives to measure false positives.
    cross_domain = [
        item
        for item in ranked_pairs
        if not item["same_domain"]
    ]
    add_candidates(
        sorted(
            cross_domain,
            key=lambda item: item["char_similarity"],
        ),
        limit=6,
    )

    # If a dataset has unusually few domain labels, fill the remainder with
    # the remaining deterministic ranking rather than silently returning less.
    add_candidates(
        sorted(
            ranked_pairs,
            key=lambda item: (
                -item["char_similarity"],
                item["pair_key"],
            ),
        )
    )
    return selected[:pair_count]


def build_annotation_rows(
    documents: list[dict[str, Any]],
    pair_count: int,
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in documents}
    ranked_pairs = _rank_pairs(documents)
    selected = _select_pairs(documents, ranked_pairs, pair_count)

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        left = documents[candidate["left_index"]]
        right = documents[candidate["right_index"]]
        left_id = str(left["id"])
        right_id = str(right["id"])
        # Keep this lookup explicit so an accidental duplicate ID cannot make
        # the generated annotation silently point at the wrong document.
        if by_id[left_id] is not left or by_id[right_id] is not right:
            raise ValueError("输入特征中存在重复文档 ID")

        if candidate["char_similarity"] >= 0.45:
            pair_type = "high_similarity_candidate"
        elif candidate["same_domain"]:
            pair_type = "same_domain_candidate"
        else:
            pair_type = "cross_domain_negative_candidate"

        rows.append(
            {
                "pair_id": f"real_pair_{index:03d}",
                "pair_type": pair_type,
                "source_type": "real_web_benchmark_v5",
                "id_a": left_id,
                "id_b": right_id,
                "title_a": left.get("title"),
                "title_b": right.get("title"),
                "domain_a": left.get("domain"),
                "domain_b": right.get("domain"),
                "language_a": left.get("language"),
                "language_b": right.get("language"),
                "char_similarity": candidate["char_similarity"],
                "text_a": left.get("clean_text", ""),
                "text_b": right.get("clean_text", ""),
                "gold_label": "",
                "label_reason": "",
                "annotation_status": "pending",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成真实网页语义去重对照集标注模板"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pair-count",
        type=int,
        default=DEFAULT_PAIR_COUNT,
    )
    args = parser.parse_args()

    documents = read_jsonl(args.input)
    rows = build_annotation_rows(documents, args.pair_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )

    print(f"真实网页候选对生成：{len(rows)} 对")
    print(f"输入文档：{len(documents)} 条")
    print(f"标签模板：{args.output}")
    print("下一步：填写每行的 gold_label 和 label_reason")


if __name__ == "__main__":
    main()
