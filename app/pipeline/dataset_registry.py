from collections import Counter

from app.pipeline.source_registry import (
    summarize_source_registry,
)


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

PROVENANCE_FIELDS = (
    "source",
    "collected_at",
    "license",
)


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


def get_missing_metadata_fields(
    document: dict,
) -> list[str]:
    missing_fields = []

    for field in PROVENANCE_FIELDS:
        value = document.get(field)

        if value is None or not str(value).strip():
            missing_fields.append(field)

    return missing_fields


def summarize_dataset(
    documents: list[dict],
    source_registry: dict | None = None,
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
    source_counts = Counter()
    license_counts = Counter()
    collected_at_counts = Counter()
    metadata_complete_count = 0

    for document in documents:
        missing_fields = get_missing_metadata_fields(
            document
        )

        for field in missing_fields:
            recommended_missing_counts[field] += 1

        if not missing_fields:
            metadata_complete_count += 1

        source = str(
            document.get("source") or ""
        ).strip() or "unknown"

        license_name = str(
            document.get("license") or ""
        ).strip() or "unknown"

        collected_at = str(
            document.get("collected_at") or ""
        ).strip() or "unknown"

        source_counts[source] += 1
        license_counts[license_name] += 1
        collected_at_counts[collected_at] += 1

    summary = {
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
        "metadata": {
            "complete_count": (
                metadata_complete_count
            ),
            "incomplete_count": (
                len(documents)
                - metadata_complete_count
            ),
            "missing_field_counts": dict(
                recommended_missing_counts
            ),
            "source_counts": dict(
                source_counts
            ),
            "license_counts": dict(
                license_counts
            ),
            "collected_at_counts": dict(
                collected_at_counts
            ),
        },
    }

    if source_registry is not None:
        summary["source_registry"] = (
            summarize_source_registry(
                documents,
                source_registry,
            )
        )

    return summary
