import json
from pathlib import Path

from tools.build_scale_fixtures import (
    build_scale_fixtures,
    iter_fixture_documents,
    parse_sizes,
)
from tools.run_performance_benchmark import benchmark_fixture


def _base_documents():
    return [
        {
            "id": "base_001",
            "url": "https://example.com/a",
            "domain": "technology",
            "language": "en",
            "source": "example.com",
            "collected_at": "2026-01-01",
            "license": "test",
            "benchmark_category": "technical_content",
            "html": (
                "<html><body><h1>Python</h1>"
                "<p>Python is a programming language.</p>"
                "<p>It supports structured data processing.</p>"
                "</body></html>"
            ),
        }
    ]


def test_parse_sizes_deduplicates_and_validates():
    assert parse_sizes("10,100,10") == (10, 100)


def test_iter_fixture_documents_are_marked_synthetic():
    documents = list(
        iter_fixture_documents(
            [
                {
                    "source_document_id": "base_001",
                    "title": "Test",
                    "html": "<p>Text</p>",
                    "url": "https://example.com",
                    "domain": "technology",
                    "language": "en",
                    "source": "example.com",
                    "collected_at": "",
                    "license": "",
                    "benchmark_category": "technical_content",
                }
            ],
            3,
            mode="unique",
        )
    )

    assert len(documents) == 3
    assert all(item["synthetic"] is True for item in documents)
    assert len({item["id"] for item in documents}) == 3
    assert "Synthetic performance variation" in documents[1]["html"]


def test_build_scale_fixtures_writes_manifest(tmp_path):
    manifest = build_scale_fixtures(
        _base_documents(),
        tmp_path,
        sizes=(3, 5),
        max_text_chars=100,
    )

    assert manifest["synthetic"] is True
    assert [item["size"] for item in manifest["fixtures"]] == [3, 5]
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["fixture_version"] == "performance_fixtures_v1"
    assert len(payload["fixtures"]) == 2


def test_benchmark_fixture_reports_all_deterministic_stages(tmp_path):
    manifest = build_scale_fixtures(
        _base_documents(),
        tmp_path,
        sizes=(4,),
        max_text_chars=100,
    )
    fixture_path = Path(manifest["fixtures"][0]["path"])
    registry = {
        "registry_version": "test_v1",
        "sources": {
            "example.com": {
                "source_name": "Example",
                "license_status": "verified",
                "usage_policy": "allow_for_experiment",
                "experiment_eligible": True,
                "release_eligible": True,
            }
        },
    }
    rules = {
        "version": "test_v1",
        "base_score": 50,
        "length": {
            "long_min_chars": 3000,
            "long_bonus": 25,
            "normal_min_chars": 100,
            "normal_bonus": 25,
            "short_min_chars": 50,
            "short_penalty": 10,
            "very_short_penalty": 20,
            "short_informative": {
                "title_body_overlap_min": 0.2,
                "repeated_line_ratio_max": 0.2,
                "removed_text_ratio_max": 0.5,
            },
            "short_informative_bonus": 10,
        },
        "repetition": {"high_ratio": 0.5, "high_penalty": 30, "medium_ratio": 0.2, "medium_penalty": 15},
        "removed_text": {"high_ratio": 0.7, "high_penalty": 25, "medium_ratio": 0.5, "medium_penalty": 10},
        "title_body_overlap": {"low_ratio": 0.2, "low_penalty": 15, "medium_ratio": 0.5, "medium_penalty": 5},
        "link_ratio": {"high_ratio": 0.4, "high_penalty": 15},
        "content_density": {"rich_bonus": 15},
        "navigation": {"score_penalty": 20},
        "aggregation": {"score_penalty": 25},
        "decision": {"keep_min_score": 70, "review_min_score": 40},
        "near_duplicate": {"threshold": 0.75},
        "score_bounds": {"min": 0, "max": 100},
    }

    report = benchmark_fixture(
        fixture_path,
        registry,
        rules,
        near_dedup_max_docs=10,
    )

    assert report["status"] == "ok"
    assert report["total"] == 4
    assert {
        stage["stage"] for stage in report["stages"]
    } == {
        "load",
        "feature_extraction",
        "exact_deduplicate",
        "near_deduplicate",
        "calibrated_scoring",
        "classification",
    }
    assert all(
        stage["status"] == "ok"
        for stage in report["stages"]
    )
