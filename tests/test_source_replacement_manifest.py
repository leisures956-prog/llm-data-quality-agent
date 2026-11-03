from tools.build_source_replacement_manifest import (
    build_replacement_manifest,
    build_replacement_template,
    select_blocked_failures,
)


def _manifest():
    return [
        {
            "id": "html_v3_001",
            "url": "https://docs.example.com/a",
            "source": "docs.example.com",
            "language": "en",
            "domain": "technology",
            "benchmark_category": "technical_content",
        },
        {
            "id": "html_v3_019",
            "url": "https://en.wikipedia.org/wiki/A",
            "source": "en.wikipedia.org",
            "language": "en",
            "domain": "technology",
            "benchmark_category": "technical_content",
        },
        {
            "id": "html_v3_024",
            "url": "https://zh.wikipedia.org/wiki/B",
            "source": "zh.wikipedia.org",
            "language": "zh",
            "domain": "technology",
            "benchmark_category": "technical_content",
        },
    ]


def _errors():
    return [
        {
            "id": "html_v3_019",
            "error_type": "HTTPStatusError",
            "error": "403 Forbidden",
        },
        {
            "id": "html_v3_024",
            "error_type": "HTTPStatusError",
            "error": "403 Forbidden",
        },
        {
            "id": "html_v3_001",
            "error_type": "ConnectError",
            "error": "connection reset",
        },
    ]


def test_select_blocked_failures_excludes_transient_errors():
    blocked, unknown = select_blocked_failures(
        _manifest(),
        _errors(),
    )

    assert [item["id"] for item in blocked] == [
        "html_v3_019",
        "html_v3_024",
    ]
    assert unknown == []


def test_replacement_template_preserves_original_metadata():
    blocked, _ = select_blocked_failures(
        _manifest(),
        _errors(),
    )
    template = build_replacement_template(blocked)

    assert len(template) == 2
    assert template[0]["replacement_for"] == (
        "html_v3_019"
    )
    assert template[0]["language"] == "en"
    assert template[1]["language"] == "zh"
    assert template[1]["original_source"] == (
        "zh.wikipedia.org"
    )


def test_build_replacement_manifest_preserves_successes():
    blocked, _ = select_blocked_failures(
        _manifest(),
        _errors(),
    )
    replacements = [
        {
            "replacement_for": "html_v3_019",
            "replacement_id": "html_v4_019",
            "url": "https://learn.example.com/ml",
            "source": "learn.example.com",
            "title": "Machine learning guide",
        },
        {
            "replacement_for": "html_v3_024",
            "replacement_id": "html_v4_024",
            "url": "https://zh.example.com/ml",
            "source": "zh.example.com",
            "title": "机器学习指南",
        },
    ]

    rows, plan = build_replacement_manifest(
        _manifest(),
        blocked,
        replacements,
    )

    assert [row["id"] for row in rows] == [
        "html_v3_001",
        "html_v4_019",
        "html_v4_024",
    ]
    assert rows[0]["url"] == (
        "https://docs.example.com/a"
    )
    assert rows[1]["replaced_from"] == "html_v3_019"
    assert rows[2]["language"] == "zh"
    assert plan["status"] == "ready"
    assert plan["blocked_count"] == 2
    assert plan["replacement_count"] == 2


def test_build_replacement_manifest_requires_all_blocked_ids():
    blocked, _ = select_blocked_failures(
        _manifest(),
        _errors(),
    )

    try:
        build_replacement_manifest(
            _manifest(),
            blocked,
            [
                {
                    "replacement_for": "html_v3_019",
                    "url": "https://learn.example.com/ml",
                }
            ],
        )
    except ValueError as exc:
        assert "html_v3_024" in str(exc)
    else:
        raise AssertionError(
            "缺少替代来源时应当失败"
        )
