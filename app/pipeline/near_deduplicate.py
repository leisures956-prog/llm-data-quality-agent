import re
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_for_similarity(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_root(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]

    return index


def _union(
    parent: list[int],
    left: int,
    right: int,
) -> None:
    left_root = _find_root(parent, left)
    right_root = _find_root(parent, right)

    if left_root != right_root:
        parent[right_root] = left_root


def annotate_near_duplicates(
    documents: list[dict],
    threshold: float = 0.75,
) -> tuple[list[dict], list[dict]]:
    if len(documents) < 2:
        return documents, []

    texts = [
        normalize_for_similarity(
            document.get("clean_text", "")
        )
        for document in documents
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
    )

    matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(matrix)

    parent = list(range(len(documents)))
    pairs = []
    max_similarity = [0.0] * len(documents)

    for left in range(len(documents)):
        for right in range(left + 1, len(documents)):
            score = float(
                similarity_matrix[left, right]
            )

            if score >= threshold:
                _union(parent, left, right)

                pairs.append({
                    "id_a": documents[left]["id"],
                    "id_b": documents[right]["id"],
                    "similarity": round(score, 4),
                })

                max_similarity[left] = max(
                    max_similarity[left],
                    score,
                )
                max_similarity[right] = max(
                    max_similarity[right],
                    score,
                )

    groups = defaultdict(list)

    for index in range(len(documents)):
        root = _find_root(parent, index)
        groups[root].append(index)

    group_ids = {}
    group_sizes = {}
    group_index = 1

    for members in groups.values():
        if len(members) > 1:
            group_id = f"near_dup_{group_index:03d}"
            group_index += 1

            for member in members:
                group_ids[member] = group_id
                group_sizes[member] = len(members)

    results = []

    for index, document in enumerate(documents):
        group_id = group_ids.get(index)

        results.append({
            **document,
            "is_near_duplicate": group_id is not None,
            "near_duplicate_group": group_id,
            "near_duplicate_count": group_sizes.get(
                index,
                1,
            ),
            "max_near_similarity": round(
                max_similarity[index],
                4,
            ),
        })

    return results, pairs