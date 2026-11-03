import json

from app.pipeline.risk_scan import (
    assess_contamination,
    assess_source_license_risk,
    build_contamination_index,
    build_risk_report,
    document_text_hash,
    scan_pii,
)


def test_scan_pii_detects_and_masks_sensitive_values():
    document = {
        "clean_text": (
            "邮箱 test.user@example.com，手机号 13800138000，"
            "身份证 11010519491231002X。"
        )
    }

    result = scan_pii(document)

    assert result["has_pii"] is True
    assert {item["type"] for item in result["matches"]} == {
        "email",
        "phone",
        "national_id",
    }
    serialized = json.dumps(result, ensure_ascii=False)
    assert "test.user@example.com" not in serialized
    assert "13800138000" not in serialized
    assert "11010519491231002X" not in serialized


def test_scan_pii_detects_labelled_name_and_address():
    result = scan_pii(
        {
            "clean_text": "姓名：张三\n通讯地址：北京市朝阳区幸福路1号"
        }
    )

    assert result["has_pii"] is True
    assert {item["type"] for item in result["matches"]} == {
        "name",
        "address",
    }


def test_scan_pii_does_not_flag_normal_technical_text():
    result = scan_pii(
        {
            "clean_text": (
                "Python lists support indexing, slicing, and iteration."
            )
        }
    )

    assert result["has_pii"] is False
    assert result["matches"] == []


def test_source_license_risk_uses_registry_statuses():
    registry = {
        "registry_version": "test_v1",
        "sources": {
            "example.com": {
                "source_name": "Example",
                "license_status": "to_verify",
                "usage_policy": "review_before_release",
                "experiment_eligible": True,
                "release_eligible": False,
            }
        },
    }

    result = assess_source_license_risk(
        {
            "source": "example.com",
            "url": "https://example.com/article",
        },
        registry,
    )

    assert result["source_registered"] is True
    assert result["experiment_eligible"] is True
    assert result["release_eligible"] is False
    assert result["license_risks"] == [
        "license_unverified",
        "release_not_eligible",
    ]


def test_contamination_matches_id_url_and_normalized_text():
    reference = [
        {
            "id": "benchmark_001",
            "url": "https://example.com/article",
            "clean_text": "Same   text.",
        }
    ]
    index = build_contamination_index(reference)

    result = assess_contamination(
        {
            "id": "benchmark_001",
            "url": "https://example.com/article#section",
            "clean_text": " SAME text. ",
        },
        index,
    )

    assert result["contaminated"] is True
    assert {
        item["type"] for item in result["matches"]
    } == {
        "benchmark_id_overlap",
        "benchmark_url_overlap",
        "benchmark_text_overlap",
    }


def test_contamination_is_skipped_without_reference():
    result = assess_contamination(
        {"id": "doc_001", "clean_text": "text"},
        None,
    )

    assert result == {
        "reference_enabled": False,
        "contaminated": False,
        "matches": [],
    }


def test_build_risk_report_keeps_decisions_unchanged():
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
    documents = [
        {
            "id": "doc_001",
            "source": "example.com",
            "url": "https://example.com/a",
            "clean_text": "A clean article.",
            "production_decision": "keep",
        }
    ]

    report = build_risk_report(documents, registry)

    assert report["risk_scan_version"] == "risk_scan_v1"
    assert report["summary"]["decision_counts"] == {"keep": 1}
    assert report["documents"][0]["decision"] == "keep"
    assert documents[0]["production_decision"] == "keep"
    assert document_text_hash(documents[0])
