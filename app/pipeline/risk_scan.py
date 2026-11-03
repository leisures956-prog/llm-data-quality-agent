"""Automated PII, source/license, and contamination risk checks.

The scanner is intentionally independent from scoring and production
decisions.  It only annotates risk evidence and produces an audit report.
Sensitive matches are masked before they are written to a report.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.pipeline.source_registry import (
    assess_document_source,
    summarize_source_registry,
)


RISK_SCAN_VERSION = "risk_scan_v1"

EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)"
)
NATIONAL_ID_RE = re.compile(
    r"(?<!\d)[1-9]\d{5}"
    r"(?:18|19|20)\d{2}"
    r"(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])"
    r"\d{3}[0-9Xx](?!\d)"
)
IPV4_RE = re.compile(
    r"(?<![\d.])"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}"
    r"(?![\d.])"
)
LABELED_NAME_RE = re.compile(
    r"(?:姓名|真实姓名|联系人|患者姓名)\s*[:：]\s*"
    r"([^\s,，。；;、]{1,20})"
)
LABELED_ADDRESS_RE = re.compile(
    r"(?:地址|住址|通讯地址|收货地址|家庭住址)\s*[:：]\s*"
    r"([^\n。；;]{2,80})"
)


def _document_text(document: dict[str, Any]) -> str:
    for field in ("clean_text", "text", "raw_text", "html"):
        value = document.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _mask_value(value: str, kind: str) -> str:
    value = str(value)
    if kind == "email" and "@" in value:
        local, domain = value.split("@", 1)
        return f"{local[:1]}***@{domain}"
    if kind == "phone":
        return f"{value[:3]}****{value[-2:]}"
    if kind == "national_id":
        return f"{value[:2]}**************{value[-2:]}"
    if kind == "ipv4":
        parts = value.split(".")
        if len(parts) == 4:
            return ".".join([parts[0], "*", "*", parts[-1]])
    if len(value) <= 2:
        return value[:1] + "*"
    return value[:1] + "*" * min(8, len(value) - 2) + value[-1:]


def _collect_pii_match(
    text: str,
    pattern: re.Pattern[str],
    pii_type: str,
    severity: str,
    group: int | None = None,
) -> dict[str, Any] | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    values = []
    for match in matches:
        value = match.group(group) if group is not None else match.group(0)
        masked = _mask_value(value, pii_type)
        if masked not in values:
            values.append(masked)

    return {
        "type": pii_type,
        "severity": severity,
        "count": len(matches),
        "masked_examples": values[:3],
    }


def scan_pii(document: dict[str, Any]) -> dict[str, Any]:
    """Scan a document for high-confidence and labelled PII patterns."""
    text = _document_text(document)
    matches = []
    patterns = (
        (EMAIL_RE, "email", "high", None),
        (PHONE_RE, "phone", "high", None),
        (NATIONAL_ID_RE, "national_id", "critical", None),
        (IPV4_RE, "ipv4", "low", None),
        (LABELED_NAME_RE, "name", "medium", 1),
        (LABELED_ADDRESS_RE, "address", "medium", 1),
    )

    for pattern, pii_type, severity, group in patterns:
        result = _collect_pii_match(
            text,
            pattern,
            pii_type,
            severity,
            group=group,
        )
        if result is not None:
            matches.append(result)

    return {
        "has_pii": bool(matches),
        "match_count": sum(item["count"] for item in matches),
        "matches": matches,
    }


def assess_source_license_risk(
    document: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Resolve source metadata and classify source/license risks."""
    assessment = assess_document_source(document, registry)
    source_risks: list[str] = []
    license_risks: list[str] = []

    if not assessment["source_registered"]:
        source_risks.append("source_unregistered")

    if assessment["usage_policy"] == "blocked":
        source_risks.append("usage_blocked")

    license_status = assessment["license_status"]
    if license_status == "unknown":
        license_risks.append("license_unknown")
    elif license_status != "verified":
        license_risks.append("license_unverified")

    if not assessment["release_eligible"]:
        license_risks.append("release_not_eligible")

    return {
        **assessment,
        "source_risks": source_risks,
        "license_risks": license_risks,
        "has_source_risk": bool(source_risks),
        "has_license_risk": bool(license_risks),
    }


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized.casefold()).strip()


def normalize_url(value: str) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""

    parsed = urlsplit(raw_value)
    if not parsed.scheme or not parsed.netloc:
        return raw_value.casefold().rstrip("/")

    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            path,
            parsed.query,
            "",
        )
    )


def document_text_hash(document: dict[str, Any]) -> str | None:
    existing_hash = str(document.get("text_hash") or "").strip()
    if existing_hash:
        return existing_hash

    text = _normalize_text(_document_text(document))
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_contamination_index(
    reference_documents: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Build ID, URL, and normalized-text indexes for a reference set."""
    index = {
        "id": {},
        "url": {},
        "text_hash": {},
    }

    for document in reference_documents:
        document_id = str(document.get("id") or "").strip()
        reference_id = document_id or "<missing-id>"

        if document_id:
            index["id"].setdefault(document_id, []).append(reference_id)

        normalized_url = normalize_url(document.get("url", ""))
        if normalized_url:
            index["url"].setdefault(normalized_url, []).append(reference_id)

        text_hash = document_text_hash(document)
        if text_hash:
            index["text_hash"].setdefault(text_hash, []).append(reference_id)

    return index


def assess_contamination(
    document: dict[str, Any],
    reference_index: dict[str, dict[str, list[str]]] | None,
) -> dict[str, Any]:
    """Find exact ID, URL, or normalized-text overlap with a reference set."""
    if not reference_index:
        return {
            "reference_enabled": False,
            "contaminated": False,
            "matches": [],
        }

    matches = []
    document_id = str(document.get("id") or "").strip()
    if document_id and document_id in reference_index["id"]:
        matches.append(
            {
                "type": "benchmark_id_overlap",
                "reference_ids": reference_index["id"][document_id],
            }
        )

    normalized_url = normalize_url(document.get("url", ""))
    if normalized_url and normalized_url in reference_index["url"]:
        matches.append(
            {
                "type": "benchmark_url_overlap",
                "reference_ids": reference_index["url"][normalized_url],
            }
        )

    text_hash = document_text_hash(document)
    if text_hash and text_hash in reference_index["text_hash"]:
        matches.append(
            {
                "type": "benchmark_text_overlap",
                "reference_ids": reference_index["text_hash"][text_hash],
            }
        )

    return {
        "reference_enabled": True,
        "contaminated": bool(matches),
        "matches": matches,
    }


def _decision(document: dict[str, Any]) -> str | None:
    for field in (
        "production_decision",
        "calibrated_decision",
        "decision",
    ):
        value = document.get(field)
        if value in {"keep", "review", "reject"}:
            return value
    return None


def scan_document(
    document: dict[str, Any],
    registry: dict[str, Any],
    reference_index: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    pii = scan_pii(document)
    source_license = assess_source_license_risk(document, registry)
    contamination = assess_contamination(document, reference_index)

    risk_flags = []
    if pii["has_pii"]:
        risk_flags.append("pii_risk")
    if source_license["has_source_risk"]:
        risk_flags.append("source_risk")
    if source_license["has_license_risk"]:
        risk_flags.append("license_risk")
    if contamination["contaminated"]:
        risk_flags.append("contamination_risk")

    return {
        "id": document.get("id"),
        "title": document.get("title", ""),
        "url": document.get("url", ""),
        "language": document.get("language", ""),
        "decision": _decision(document),
        "pii": pii,
        "source_license": source_license,
        "contamination": contamination,
        "risk_flags": risk_flags,
        "has_any_risk": bool(risk_flags),
    }


def build_risk_report(
    documents: list[dict[str, Any]],
    registry: dict[str, Any],
    reference_documents: list[dict[str, Any]] | None = None,
    input_paths: dict[str, str | None] | None = None,
    reference_path: str | None = None,
) -> dict[str, Any]:
    reference_documents = reference_documents or []
    reference_index = (
        build_contamination_index(reference_documents)
        if reference_documents
        else None
    )
    rows = [
        scan_document(document, registry, reference_index)
        for document in documents
    ]

    risk_flag_counts = Counter(
        flag
        for row in rows
        for flag in row["risk_flags"]
    )
    pii_type_counts = Counter(
        match["type"]
        for row in rows
        for match in row["pii"]["matches"]
    )
    source_risk_counts = Counter(
        risk
        for row in rows
        for risk in row["source_license"]["source_risks"]
    )
    license_risk_counts = Counter(
        risk
        for row in rows
        for risk in row["source_license"]["license_risks"]
    )
    contamination_match_counts = Counter(
        match["type"]
        for row in rows
        for match in row["contamination"]["matches"]
    )

    summary = {
        "total": len(rows),
        "documents_with_any_risk": sum(
            row["has_any_risk"] for row in rows
        ),
        "pii_risk_document_count": sum(
            row["pii"]["has_pii"] for row in rows
        ),
        "source_risk_document_count": sum(
            row["source_license"]["has_source_risk"]
            for row in rows
        ),
        "license_risk_document_count": sum(
            row["source_license"]["has_license_risk"]
            for row in rows
        ),
        "contamination_document_count": sum(
            row["contamination"]["contaminated"]
            for row in rows
        ),
        "risk_flag_counts": dict(sorted(risk_flag_counts.items())),
        "pii_type_counts": dict(sorted(pii_type_counts.items())),
        "source_risk_counts": dict(sorted(source_risk_counts.items())),
        "license_risk_counts": dict(sorted(license_risk_counts.items())),
        "contamination_match_counts": dict(
            sorted(contamination_match_counts.items())
        ),
        "decision_counts": dict(
            sorted(
                Counter(
                    row["decision"]
                    for row in rows
                    if row["decision"]
                ).items()
            )
        ),
    }

    return {
        "risk_scan_version": RISK_SCAN_VERSION,
        "input": input_paths or {},
        "reference": {
            "enabled": bool(reference_documents),
            "path": reference_path,
            "count": len(reference_documents),
        },
        "source_registry": summarize_source_registry(
            documents,
            registry,
        ),
        "summary": summary,
        "documents": rows,
    }
