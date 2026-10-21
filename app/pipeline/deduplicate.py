import hashlib
import re
from collections import defaultdict


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_hash(text: str) -> str:
    normalized_text = normalize_text(text)

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def annotate_exact_duplicates(
    documents: list[dict],
) -> list[dict]:
    groups = defaultdict(list)
    fingerprints = {}

    for document in documents:
        document_id = document["id"]
        fingerprint = text_hash(
            document.get("clean_text", "")
        )

        fingerprints[document_id] = fingerprint
        groups[fingerprint].append(document_id)

    group_info = {}
    group_index = 1

    for fingerprint, document_ids in groups.items():
        is_duplicate = len(document_ids) > 1

        if is_duplicate:
            group_id = f"dup_{group_index:03d}"
            group_index += 1
        else:
            group_id = None

        for document_id in document_ids:
            group_info[document_id] = {
                "is_duplicate": is_duplicate,
                "duplicate_group": group_id,
                "duplicate_count": len(document_ids),
            }

    results = []

    for document in documents:
        document_id = document["id"]
        info = group_info[document_id]

        results.append({
            **document,
            "text_hash": fingerprints[document_id],
            **info,
        })

    return results

def select_unique_documents(
    documents: list[dict],
) -> list[dict]:
    unique_documents = []
    kept_groups = set()

    for document in documents:
        duplicate_group = document.get(
            "duplicate_group"
        )

        # 没有重复组，直接保留
        if duplicate_group is None:
            unique_documents.append(document)
            continue

        # 每个重复组只保留第一条
        if duplicate_group not in kept_groups:
            unique_documents.append(document)
            kept_groups.add(duplicate_group)

    return unique_documents