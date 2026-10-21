import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_SOURCE_REGISTRY_PATH = Path(
    "config/source_registry.json"
)

LICENSE_STATUSES = {
    "verified",
    "to_verify",
    "restricted",
    "unknown",
}

USAGE_POLICIES = {
    "allow_for_experiment",
    "review_before_release",
    "blocked",
    "unregistered",
}


def normalize_source_key(value: str) -> str:
    """Normalize a source domain or URL to a registry key."""
    raw_value = str(value or "").strip().casefold()

    if not raw_value:
        return ""

    parsed = urlparse(
        raw_value
        if "://" in raw_value
        else f"//{raw_value}",
        allow_fragments=True,
    )

    host = parsed.hostname or ""

    if not host:
        host = raw_value.split("/", 1)[0]

    host = host.strip().rstrip(".")

    if host.startswith("www."):
        host = host[4:]

    return host


def _registry_sources(registry: dict) -> dict:
    sources = registry.get("sources", registry)

    if not isinstance(sources, dict):
        raise ValueError(
            "来源注册表的 sources 必须是对象"
        )

    return sources


def load_source_registry(
    path: Path = DEFAULT_SOURCE_REGISTRY_PATH,
) -> dict:
    """Load and validate the source registry configuration."""
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "来源注册表必须是 JSON 对象"
        )

    sources = _registry_sources(payload)
    normalized_sources = {}

    for source_key, record in sources.items():
        if not isinstance(record, dict):
            raise ValueError(
                f"来源记录必须是对象：{source_key}"
            )

        normalized_key = normalize_source_key(
            source_key
        )

        if not normalized_key:
            raise ValueError(
                f"来源注册表存在空来源键：{source_key}"
            )

        license_status = str(
            record.get("license_status", "unknown")
        ).strip().casefold()

        if license_status not in LICENSE_STATUSES:
            raise ValueError(
                "不支持的 license_status："
                f"{license_status}（来源：{source_key}）"
            )

        usage_policy = str(
            record.get(
                "usage_policy",
                "review_before_release",
            )
        ).strip().casefold()

        if usage_policy not in USAGE_POLICIES:
            raise ValueError(
                "不支持的 usage_policy："
                f"{usage_policy}（来源：{source_key}）"
            )

        normalized_sources[normalized_key] = {
            **record,
            "source_key": normalized_key,
            "license_status": license_status,
            "usage_policy": usage_policy,
        }

    return {
        "registry_version": str(
            payload.get(
                "registry_version",
                "source_registry_v1",
            )
        ),
        "sources": normalized_sources,
    }


def resolve_document_source(
    document: dict,
) -> tuple[str, str]:
    """Return a normalized source key and how it was resolved."""
    declared_source = str(
        document.get("source") or ""
    ).strip()

    if declared_source:
        return (
            normalize_source_key(declared_source),
            "source_field",
        )

    url = str(document.get("url") or "").strip()
    url_source = normalize_source_key(url)

    if url_source:
        return url_source, "url_fallback"

    return "unknown", "missing"


def assess_document_source(
    document: dict,
    registry: dict,
) -> dict:
    source_key, resolution = resolve_document_source(
        document
    )
    record = _registry_sources(registry).get(
        source_key
    )

    if record is None:
        return {
            "source_key": source_key or "unknown",
            "source_resolution": resolution,
            "source_registered": False,
            "source_registry_status": "unregistered",
            "source_registry_name": "unknown",
            "license_status": "unknown",
            "usage_policy": "unregistered",
            "experiment_eligible": False,
            "release_eligible": False,
        }

    license_status = record["license_status"]
    usage_policy = record["usage_policy"]

    experiment_eligible = bool(
        record.get(
            "experiment_eligible",
            license_status in {
                "verified",
                "to_verify",
            },
        )
    )
    release_eligible = bool(
        record.get(
            "release_eligible",
            license_status == "verified"
            and usage_policy
            == "allow_for_experiment",
        )
    )

    return {
        "source_key": source_key,
        "source_resolution": resolution,
        "source_registered": True,
        "source_registry_status": "registered",
        "source_registry_name": record.get(
            "source_name",
            source_key,
        ),
        "license_status": license_status,
        "usage_policy": usage_policy,
        "experiment_eligible": experiment_eligible,
        "release_eligible": release_eligible,
    }


def annotate_document_source(
    document: dict,
    registry: dict,
) -> dict:
    return {
        **document,
        "source_registry_version": registry[
            "registry_version"
        ],
        **assess_document_source(
            document,
            registry,
        ),
    }


def summarize_source_registry(
    documents: list[dict],
    registry: dict,
) -> dict:
    registered_count = 0
    experiment_eligible_count = 0
    release_eligible_count = 0
    source_counts = Counter()
    license_status_counts = Counter()
    usage_policy_counts = Counter()
    resolution_counts = Counter()
    unregistered_sources = set()

    for document in documents:
        assessment = assess_document_source(
            document,
            registry,
        )

        source_key = assessment["source_key"]
        source_counts[source_key] += 1
        license_status_counts[
            assessment["license_status"]
        ] += 1
        usage_policy_counts[
            assessment["usage_policy"]
        ] += 1
        resolution_counts[
            assessment["source_resolution"]
        ] += 1

        if assessment["source_registered"]:
            registered_count += 1
        else:
            unregistered_sources.add(source_key)

        if assessment["experiment_eligible"]:
            experiment_eligible_count += 1

        if assessment["release_eligible"]:
            release_eligible_count += 1

    total = len(documents)

    return {
        "registry_version": registry[
            "registry_version"
        ],
        "registered_count": registered_count,
        "unregistered_count": total - registered_count,
        "registered_rate": round(
            registered_count / max(total, 1),
            4,
        ),
        "experiment_eligible_count": (
            experiment_eligible_count
        ),
        "release_eligible_count": (
            release_eligible_count
        ),
        "source_counts": dict(source_counts),
        "license_status_counts": dict(
            license_status_counts
        ),
        "usage_policy_counts": dict(
            usage_policy_counts
        ),
        "source_resolution_counts": dict(
            resolution_counts
        ),
        "unregistered_sources": sorted(
            source
            for source in unregistered_sources
            if source
        ),
    }
