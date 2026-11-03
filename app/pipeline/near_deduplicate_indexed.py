"""Experimental indexed candidate generation for near deduplication.

This module keeps the existing character n-gram cosine threshold unchanged.
It only avoids scoring document pairs that share no n-gram feature.  The
production near-deduplication implementation is intentionally untouched.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_THRESHOLD = 0.75
DEFAULT_NGRAM_RANGE = (3, 5)


def normalize_for_similarity(text: str) -> str:
    text = str(text or "").casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _validate_threshold(threshold: float) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须位于 0 到 1 之间")


def _find_root(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find_root(parent, left)
    right_root = _find_root(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _build_candidate_pairs(
    matrix: Any,
) -> set[tuple[int, int]]:
    """Return pairs represented by non-zero entries in a sparse matrix.

    The caller passes the sparse document-document similarity matrix.  Using
    its coordinate lists avoids Python-level combinations over every frequent
    n-gram posting list, which can become disproportionately expensive on
    synthetic fixtures where many documents share the same boilerplate.
    """
    rows, columns = matrix.nonzero()
    return {
        (int(left), int(right))
        for left, right in zip(rows, columns)
        if left < right
    }


def _annotate_results(
    documents: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    max_similarity: list[float],
) -> list[dict[str, Any]]:
    parent = list(range(len(documents)))
    index_by_id = {
        document["id"]: index
        for index, document in enumerate(documents)
    }

    for pair in pairs:
        _union(
            parent,
            index_by_id[pair["id_a"]],
            index_by_id[pair["id_b"]],
        )

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(documents)):
        groups[_find_root(parent, index)].append(index)

    group_ids: dict[int, str] = {}
    group_sizes: dict[int, int] = {}
    group_index = 1
    for members in groups.values():
        if len(members) <= 1:
            continue
        group_id = f"near_dup_indexed_{group_index:03d}"
        group_index += 1
        for member in members:
            group_ids[member] = group_id
            group_sizes[member] = len(members)

    results = []
    for index, document in enumerate(documents):
        group_id = group_ids.get(index)
        results.append(
            {
                **document,
                "is_near_duplicate": group_id is not None,
                "near_duplicate_group": group_id,
                "near_duplicate_count": group_sizes.get(index, 1),
                "max_near_similarity": round(max_similarity[index], 4),
            }
        )
    return results


def annotate_near_duplicates_indexed(
    documents: list[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run indexed candidate generation with the existing cosine threshold."""
    _validate_threshold(threshold)
    document_count = len(documents)
    theoretical_pairs = document_count * max(document_count - 1, 0) // 2
    if document_count < 2:
        results = _annotate_results(documents, [], [0.0] * document_count)
        return (
            results,
            [],
            {
                "theoretical_pair_count": theoretical_pairs,
                "candidate_pair_count": 0,
                "scored_pair_count": 0,
                "candidate_reduction_ratio": 1.0,
                "vocabulary_size": 0,
            },
        )

    texts = [
        normalize_for_similarity(document.get("clean_text", ""))
        for document in documents
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    similarity_matrix = matrix @ matrix.T
    candidate_pairs = _build_candidate_pairs(similarity_matrix)
    ordered_candidates = sorted(candidate_pairs)
    pairs = []
    max_similarity = [0.0] * document_count

    for left, right in ordered_candidates:
        score = float(similarity_matrix[left, right])
        if score < threshold:
            continue
        pairs.append(
            {
                "id_a": documents[left]["id"],
                "id_b": documents[right]["id"],
                "similarity": round(score, 4),
            }
        )
        max_similarity[left] = max(max_similarity[left], score)
        max_similarity[right] = max(max_similarity[right], score)

    candidate_count = len(ordered_candidates)
    reduction_ratio = 1.0
    if theoretical_pairs:
        reduction_ratio = 1 - candidate_count / theoretical_pairs

    stats = {
        "theoretical_pair_count": theoretical_pairs,
        "candidate_pair_count": candidate_count,
        "scored_pair_count": candidate_count,
        "candidate_reduction_ratio": round(reduction_ratio, 4),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }
    return _annotate_results(documents, pairs, max_similarity), pairs, stats
