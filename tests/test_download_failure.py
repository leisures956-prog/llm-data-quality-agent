from tools.download_failure import (
    classify_download_failure,
)
from tools.retry_html_benchmark import (
    select_retryable_items,
)


def test_network_failure_is_retryable():
    result = classify_download_failure(
        "ConnectError",
        "[SSL: UNEXPECTED_EOF_WHILE_READING]",
    )

    assert result["error_class"] == "transient_network"
    assert result["retryable"] is True
    assert result["suggested_action"] == "retry_with_backoff"


def test_blocked_http_status_is_not_retryable():
    result = classify_download_failure(
        "HTTPStatusError",
        "Client error '403 Forbidden' for url",
    )

    assert result["error_class"] == "blocked_or_unavailable"
    assert result["retryable"] is False
    assert result["status_code"] == 403


def test_transient_http_status_is_retryable():
    result = classify_download_failure(
        "HTTPStatusError",
        "Server error '503 Service Unavailable' for url",
    )

    assert result["error_class"] == "transient_http"
    assert result["retryable"] is True
    assert result["status_code"] == 503


def test_retry_selection_uses_manifest_and_skips_blocked_errors():
    manifest = [
        {"id": "doc_network", "url": "https://example.com/a"},
        {"id": "doc_blocked", "url": "https://example.com/b"},
    ]
    errors = [
        {
            "id": "doc_network",
            "error_type": "ConnectError",
            "error": "connection reset",
        },
        {
            "id": "doc_blocked",
            "error_type": "HTTPStatusError",
            "error": "403 Forbidden",
        },
    ]

    selected, missing = select_retryable_items(
        manifest,
        errors,
    )

    assert [item["id"] for item in selected] == [
        "doc_network"
    ]
    assert missing == []
