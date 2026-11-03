"""Embedding-based semantic duplicate detection.

The model dependency is optional at import time.  Production runs use a
SentenceTransformers encoder, while tests can inject a small deterministic
encoder so the grouping logic remains fast and reproducible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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


def _validate_threshold(threshold: float) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError(
            f"semantic threshold 必须在 0 到 1 之间：{threshold}"
        )


def load_sentence_transformer(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load the optional SentenceTransformers model lazily."""

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少 sentence-transformers。请先安装 "
            "requirements-benchmark.txt，或在测试中注入 encoder。"
        ) from exc

    return SentenceTransformer(model_name)


def _encode(
    texts: list[str],
    encoder: Any,
    batch_size: int,
) -> Any:
    if hasattr(encoder, "encode"):
        return encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    if callable(encoder):
        return encoder(texts)
    raise TypeError("encoder 必须提供 encode() 方法或可调用接口")


def _build_group_info(
    documents: list[dict],
    pairs: list[dict],
) -> tuple[dict[int, str], dict[int, int]]:
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

    groups = defaultdict(list)
    for index in range(len(documents)):
        groups[_find_root(parent, index)].append(index)

    group_ids: dict[int, str] = {}
    group_sizes: dict[int, int] = {}
    group_index = 1
    for members in groups.values():
        if len(members) <= 1:
            continue
        group_id = f"semantic_dup_{group_index:03d}"
        group_index += 1
        for member in members:
            group_ids[member] = group_id
            group_sizes[member] = len(members)

    return group_ids, group_sizes


def annotate_semantic_duplicates(
    documents: list[dict],
    threshold: float = 0.80,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Any | None = None,
    batch_size: int = 32,
) -> tuple[list[dict], list[dict]]:
    """Annotate semantic duplicate groups using embedding cosine similarity.

    ``encoder`` is injectable for tests.  If it is omitted, a multilingual
    SentenceTransformers model is loaded lazily.
    """

    _validate_threshold(threshold)
    if len(documents) < 2:
        return [
            {
                **document,
                "is_semantic_duplicate": False,
                "semantic_duplicate_group": None,
                "semantic_duplicate_count": 1,
                "max_semantic_similarity": 0.0,
            }
            for document in documents
        ], []

    if encoder is None:
        encoder = load_sentence_transformer(model_name)

    texts = [
        str(document.get("clean_text", "") or "")
        for document in documents
    ]
    embeddings = _encode(texts, encoder, batch_size=batch_size)
    similarity_matrix = cosine_similarity(embeddings)

    pairs = []
    max_similarity = [0.0] * len(documents)
    for left in range(len(documents)):
        for right in range(left + 1, len(documents)):
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
            max_similarity[left] = max(
                max_similarity[left],
                score,
            )
            max_similarity[right] = max(
                max_similarity[right],
                score,
            )

    group_ids, group_sizes = _build_group_info(
        documents,
        pairs,
    )
    results = []
    for index, document in enumerate(documents):
        group_id = group_ids.get(index)
        results.append(
            {
                **document,
                "is_semantic_duplicate": group_id is not None,
                "semantic_duplicate_group": group_id,
                "semantic_duplicate_count": group_sizes.get(
                    index,
                    1,
                ),
                "max_semantic_similarity": round(
                    max_similarity[index],
                    4,
                ),
            }
        )

    return results, pairs
