from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import httpx
from bs4 import BeautifulSoup

from tools.download_failure import (
    classify_download_failure,
    classify_error_record,
)
from tools.download_html_benchmark import HEADERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = resolve_path(path)
    rows = []

    for line_number, line in enumerate(
        resolved_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        item = json.loads(line)

        if not isinstance(item, dict):
            raise ValueError(
                f"{resolved_path}:{line_number} 不是 JSON 对象"
            )

        rows.append(item)

    return rows


def write_jsonl(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    resolved_path = resolve_path(path)
    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    resolved_path.write_text(
        "\n".join(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            for item in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def select_retryable_items(
    manifest_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_map = {}

    for item in manifest_rows:
        item_id = str(item["id"])

        if item_id in manifest_map:
            raise ValueError(
                f"发现重复清单 ID：{item_id}"
            )

        manifest_map[item_id] = item

    selected = []
    selected_ids = set()
    missing_manifest_ids = []

    for error in error_rows:
        failure = classify_error_record(error)

        if not failure["retryable"]:
            continue

        item_id = str(error.get("id") or "")

        if item_id not in manifest_map:
            missing_manifest_ids.append(item_id)
            continue

        if item_id in selected_ids:
            continue

        selected.append(manifest_map[item_id])
        selected_ids.add(item_id)

    return selected, sorted(
        set(missing_manifest_ids)
    )


def fetch_item(
    client: httpx.Client,
    item: dict[str, Any],
) -> dict[str, Any]:
    url = str(item["url"])
    response = client.get(url)
    response.raise_for_status()

    html = response.text

    if len(html.strip()) < 200:
        raise ValueError("HTML 内容过短")

    soup = BeautifulSoup(
        html,
        "html.parser",
    )
    title = ""

    if soup.title is not None:
        title = soup.title.get_text(
            " ",
            strip=True,
        )

    if not title:
        title = item.get(
            "title",
            url,
        )

    return {
        "id": item["id"],
        "url": str(response.url),
        "domain": item.get(
            "domain",
            "unknown",
        ),
        "language": item.get(
            "language",
            "unknown",
        ),
        "source": item.get(
            "source",
            urlparse(url).netloc,
        ),
        "collected_at": datetime.now(
            timezone.utc
        ).date().isoformat(),
        "license": item.get(
            "license",
            "page_terms_to_verify",
        ),
        "benchmark_category": item.get(
            "benchmark_category",
            "",
        ),
        "title": title,
        "html": html,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只重试可恢复的 HTML 下载失败"
    )
    parser.add_argument(
        "--manifest",
        default="data/raw/html_urls_v3.jsonl",
        help="原始 URL 清单",
    )
    parser.add_argument(
        "--base",
        default="data/raw/html_benchmark_v3.jsonl",
        help="已有成功数据文件",
    )
    parser.add_argument(
        "--errors",
        default="data/raw/html_benchmark_v3_errors.jsonl",
        help="原始下载错误文件",
    )
    parser.add_argument(
        "--output",
        default="data/raw/html_benchmark_v3_retry_v1.jsonl",
        help="合并后的新数据文件",
    )
    parser.add_argument(
        "--retry-errors",
        default="data/raw/html_benchmark_v3_retry_v1_errors.jsonl",
        help="重试后的错误文件",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="每次请求之间的间隔秒数",
    )

    args = parser.parse_args()
    output_path = resolve_path(args.output)
    retry_error_path = resolve_path(args.retry_errors)

    if output_path.exists():
        raise FileExistsError(
            f"输出文件已存在，为避免覆盖：{output_path}"
        )

    if retry_error_path.exists():
        raise FileExistsError(
            "重试错误文件已存在，为避免覆盖："
            f"{retry_error_path}"
        )

    manifest_rows = load_jsonl(args.manifest)
    base_rows = load_jsonl(args.base)
    error_rows = load_jsonl(args.errors)

    base_map = {}

    for item in base_rows:
        item_id = str(item["id"])

        if item_id in base_map:
            raise ValueError(
                f"已有数据中存在重复 ID：{item_id}"
            )

        base_map[item_id] = item

    retry_items, missing_manifest_ids = (
        select_retryable_items(
            manifest_rows,
            error_rows,
        )
    )

    recovered = {}
    retry_failures = []

    timeout = httpx.Timeout(30.0)

    with httpx.Client(
        headers=HEADERS,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        for index, item in enumerate(
            retry_items,
            start=1,
        ):
            item_id = str(item["id"])
            print(
                f"[{index}/{len(retry_items)}] 重试："
                f"{item['url']}"
            )

            if item_id in base_map:
                print("  已存在于成功数据，跳过")
                continue

            try:
                recovered[item_id] = fetch_item(
                    client,
                    item,
                )
            except Exception as exc:
                response = getattr(
                    exc,
                    "response",
                    None,
                )
                status_code = getattr(
                    response,
                    "status_code",
                    None,
                )
                failure = classify_download_failure(
                    type(exc).__name__,
                    str(exc),
                    status_code,
                )
                retry_failures.append(
                    {
                        "id": item_id,
                        "url": item["url"],
                        "error_type": type(
                            exc
                        ).__name__,
                        "error": str(exc),
                        "retry_attempted": True,
                        **failure,
                    }
                )
                print(
                    f"  重试失败：{type(exc).__name__}"
                )

            if args.delay > 0:
                time.sleep(args.delay)

    combined_map = {
        **base_map,
        **recovered,
    }
    manifest_ids = [
        str(item["id"])
        for item in manifest_rows
    ]
    ordered_ids = [
        item_id
        for item_id in manifest_ids
        if item_id in combined_map
    ]
    extra_ids = sorted(
        set(combined_map) - set(ordered_ids)
    )
    combined_rows = [
        combined_map[item_id]
        for item_id in ordered_ids + extra_ids
    ]

    retry_ids = {
        str(item["id"])
        for item in retry_items
    }
    recovered_ids = set(recovered)
    final_errors = []

    for error in error_rows:
        item_id = str(error.get("id") or "")

        if item_id in recovered_ids:
            continue

        failure = classify_error_record(error)
        normalized = {
            **error,
            **failure,
            "retry_attempted": item_id in retry_ids,
        }
        final_errors.append(normalized)

    final_errors.extend(retry_failures)

    write_jsonl(output_path, combined_rows)
    write_jsonl(retry_error_path, final_errors)

    print(f"可重试失败：{len(retry_items)} 条")
    print(f"重试成功：{len(recovered)} 条")
    print(f"重试后仍失败：{len(retry_failures)} 条")
    print(
        f"清单中找不到的失败 ID："
        f"{len(missing_manifest_ids)} 条"
    )
    print(f"合并数据：{output_path}")
    print(f"重试错误：{retry_error_path}")


if __name__ == "__main__":
    main()
