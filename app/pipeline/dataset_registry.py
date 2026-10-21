from collections import Counter


REQUIRED_FIELDS = {
    "id",
    "url",
    "domain",
    "language",
    "html",
}

RECOMMENDED_FIELDS = {
    "source",
    "collected_at",
    "license",
}


def validate_raw_document(
    document: dict,
    line_number: int,
) -> list[str]:
    errors = []

    if not isinstance(document, dict):
        return [
            f"第 {line_number} 行不是 JSON 对象"
        ]

    for field in REQUIRED_FIELDS:
        if field not in document:
            errors.append(
                f"缺少必需字段：{field}"
            )

    document_id = document.get("id")

    if not isinstance(document_id, str) or not document_id.strip():
        errors.append("id 不能为空")

    html = document.get("html", "")

    if not isinstance(html, str):
        errors.append("html 必须是字符串")
    elif len(html.strip()) < 20:
        errors.append("html 内容过短")

    return errors


def summarize_dataset(
    documents: list[dict],
) -> dict:
    id_counts = Counter(
        item.get("id", "")
        for item in documents
    )

    language_counts = Counter(
        item.get("language", "unknown")
        for item in documents
    )

    domain_counts = Counter(
        item.get("domain", "unknown")
        for item in documents
    )

    duplicate_ids = sorted(
        document_id
        for document_id, count in id_counts.items()
        if document_id and count > 1
    )

    recommended_missing_counts = Counter()

    for document in documents:
        for field in RECOMMENDED_FIELDS:
            value = document.get(field)

            if value is None or not str(value).strip():
                recommended_missing_counts[field] += 1

    return {
        "total": len(documents),
        "language_counts": dict(
            language_counts
        ),
        "domain_counts": dict(
            domain_counts
        ),
        "duplicate_ids": duplicate_ids,
        "recommended_metadata_missing": dict(
            recommended_missing_counts
        ),
    }