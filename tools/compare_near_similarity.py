import json
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


input_path = Path("data/processed/unique.jsonl")

documents = [
    json.loads(line)
    for line in input_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

language_groups = defaultdict(list)

for document in documents:
    text = document.get("clean_text", "").strip()

    if text:
        language_groups[
            document.get("language", "unknown")
        ].append(document)


configs = [
    (
        "char_2_5",
        {
            "analyzer": "char",
            "ngram_range": (2, 5),
        },
    ),
    (
        "char_3_5",
        {
            "analyzer": "char",
            "ngram_range": (3, 5),
        },
    ),
    (
        "word_1_2",
        {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "token_pattern": r"(?u)\b\w+\b",
        },
    ),
]


for config_name, vectorizer_kwargs in configs:
    pairs = []

    for language, group in language_groups.items():
        if len(group) < 2:
            continue

        texts = [
            item.get("clean_text", "").casefold()
            for item in group
        ]

        vectorizer = TfidfVectorizer(
            min_df=1,
            **vectorizer_kwargs,
        )

        matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(matrix)

        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                pairs.append(
                    {
                        "language": language,
                        "id_a": group[left]["id"],
                        "title_a": group[left].get("title", ""),
                        "id_b": group[right]["id"],
                        "title_b": group[right].get("title", ""),
                        "similarity": float(
                            similarity_matrix[left, right]
                        ),
                    }
                )

    pairs.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    print(f"\n===== {config_name} =====")
    print(f"候选对总数：{len(pairs)}")

    for threshold in (0.75, 0.80, 0.85):
        count = sum(
            item["similarity"] >= threshold
            for item in pairs
        )
        print(
            f"相似度 >= {threshold}：{count}"
        )

    print("最高相似度的前 10 组：")

    for item in pairs[:10]:
        print(
            f'{item["id_a"]} ({item["title_a"]}) '
            f'<-> {item["id_b"]} ({item["title_b"]}) '
            f': {item["similarity"]:.4f}'
        )

    target_pair = {
        "doc_001",
        "doc_007",
    }

    for item in pairs:
        current_pair = {
            item["id_a"],
            item["id_b"],
        }

        if current_pair == target_pair:
            print(
                "doc_001/doc_007 相似度："
                f'{item["similarity"]:.4f}'
            )
            break