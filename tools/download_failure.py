from __future__ import annotations

import re
from typing import Any


RETRYABLE_ERROR_TYPES = {
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
    "TimeoutException",
    "NetworkError",
}

RETRYABLE_STATUS_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}

BLOCKED_STATUS_CODES = {
    401,
    403,
    404,
    410,
    451,
}


def extract_status_code(
    error_message: str,
) -> int | None:
    match = re.search(
        r"\b([1-5][0-9]{2})\b",
        str(error_message),
    )

    if match is None:
        return None

    return int(match.group(1))


def classify_download_failure(
    error_type: str | None,
    error_message: str | None = None,
    status_code: int | None = None,
) -> dict[str, Any]:
    normalized_type = str(
        error_type or "UnknownError"
    )
    message = str(error_message or "")

    if status_code is None:
        status_code = extract_status_code(message)

    if status_code in RETRYABLE_STATUS_CODES:
        return {
            "error_class": "transient_http",
            "retryable": True,
            "suggested_action": "retry_with_backoff",
            "status_code": status_code,
        }

    if status_code in BLOCKED_STATUS_CODES:
        return {
            "error_class": "blocked_or_unavailable",
            "retryable": False,
            "suggested_action": "replace_source_or_verify_access_policy",
            "status_code": status_code,
        }

    if normalized_type in RETRYABLE_ERROR_TYPES:
        return {
            "error_class": "transient_network",
            "retryable": True,
            "suggested_action": "retry_with_backoff",
            "status_code": status_code,
        }

    if normalized_type == "HTTPStatusError":
        return {
            "error_class": "http_error",
            "retryable": False,
            "suggested_action": "inspect_status_and_source_policy",
            "status_code": status_code,
        }

    if normalized_type in {
        "JSONDecodeError",
        "ValueError",
    }:
        return {
            "error_class": "invalid_response",
            "retryable": False,
            "suggested_action": "inspect_response_or_parser",
            "status_code": status_code,
        }

    return {
        "error_class": "unknown_error",
        "retryable": False,
        "suggested_action": "inspect_manually",
        "status_code": status_code,
    }


def classify_error_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    return classify_download_failure(
        record.get("error_type"),
        record.get("error"),
        record.get("status_code"),
    )
