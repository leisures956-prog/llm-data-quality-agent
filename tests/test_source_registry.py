import json

import pytest

from app.pipeline.source_registry import (
    assess_document_source,
    load_source_registry,
    normalize_source_key,
    summarize_source_registry,
)


def test_normalize_source_key_accepts_domain_and_url():
    assert normalize_source_key(
        "https://WWW.Example.com/path"
    ) == "example.com"


def test_load_source_registry_validates_statuses(
    tmp_path,
):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "registry_version": "test_v1",
                "sources": {
                    "example.com": {
                        "source_name": "Example",
                        "license_status": "to_verify",
                        "usage_policy": (
                            "review_before_release"
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    registry = load_source_registry(path)

    assert registry["registry_version"] == "test_v1"
    assert "example.com" in registry["sources"]


def test_load_source_registry_rejects_unknown_status(
    tmp_path,
):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "example.com": {
                        "license_status": "invalid"
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_source_registry(path)


def test_assess_document_source_uses_url_fallback():
    registry = {
        "registry_version": "test_v1",
        "sources": {
            "example.com": {
                "source_name": "Example",
                "license_status": "to_verify",
                "usage_policy": (
                    "review_before_release"
                ),
                "experiment_eligible": True,
                "release_eligible": False,
            }
        },
    }

    result = assess_document_source(
        {
            "url": "https://www.example.com/article"
        },
        registry,
    )

    assert result["source_key"] == "example.com"
    assert result["source_resolution"] == (
        "url_fallback"
    )
    assert result["source_registered"] is True
    assert result["experiment_eligible"] is True
    assert result["release_eligible"] is False


def test_summarize_source_registry_counts_unknown_sources():
    registry = {
        "registry_version": "test_v1",
        "sources": {
            "example.com": {
                "source_name": "Example",
                "license_status": "verified",
                "usage_policy": (
                    "allow_for_experiment"
                ),
                "experiment_eligible": True,
                "release_eligible": True,
            }
        },
    }

    summary = summarize_source_registry(
        [
            {
                "source": "example.com",
                "url": "https://example.com/a",
            },
            {
                "source": "unknown.org",
                "url": "https://unknown.org/b",
            },
        ],
        registry,
    )

    assert summary["registered_count"] == 1
    assert summary["unregistered_count"] == 1
    assert summary["experiment_eligible_count"] == 1
    assert summary["release_eligible_count"] == 1
    assert summary["unregistered_sources"] == [
        "unknown.org"
    ]
