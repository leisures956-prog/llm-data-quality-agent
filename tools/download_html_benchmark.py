import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from tools.download_failure import (
    classify_download_failure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data/raw/html_urls_v2.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data/raw/html_benchmark_v2.jsonl"
)
DEFAULT_ERROR_PATH = (
    PROJECT_ROOT / "data/raw/html_benchmark_v2_errors.jsonl"
)

# 保留旧常量名称，避免已有本地脚本导入时发生不兼容。
MANIFEST_PATH = DEFAULT_MANIFEST_PATH
OUTPUT_PATH = DEFAULT_OUTPUT_PATH
ERROR_PATH = DEFAULT_ERROR_PATH

HEADERS = {
    "User-Agent": (
        "llm-data-quality-benchmark/0.1 "
        "(research dataset quality evaluation)"
    )
}


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def resolve_configured_path(
    env_name,
    default_path,
):
    value = os.environ.get(env_name, "").strip()

    if not value:
        return default_path

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def get_download_paths():
    """Return manifest/output/error paths selected by the environment.

    Relative paths are resolved from the project root, so the downloader
    remains reproducible regardless of the caller's current directory.
    """

    return {
        "manifest": resolve_configured_path(
            "HTML_MANIFEST_PATH",
            DEFAULT_MANIFEST_PATH,
        ),
        "output": resolve_configured_path(
            "HTML_OUTPUT_PATH",
            DEFAULT_OUTPUT_PATH,
        ),
        "errors": resolve_configured_path(
            "HTML_ERROR_PATH",
            DEFAULT_ERROR_PATH,
        ),
    }


def main():
    paths = get_download_paths()
    manifest_path = paths["manifest"]
    output_path = paths["output"]
    error_path = paths["errors"]

    if output_path.exists():
        raise FileExistsError(
            f"输出文件已存在，为避免覆盖：{output_path}"
        )

    manifest = read_jsonl(manifest_path)

    ids = [item["id"] for item in manifest]
    urls = [item["url"] for item in manifest]

    if len(ids) != len(set(ids)):
        raise ValueError("URL 清单中存在重复 ID")

    if len(urls) != len(set(urls)):
        raise ValueError("URL 清单中存在重复 URL")

    records = []
    errors = []

    timeout = httpx.Timeout(30.0)

    with httpx.Client(
        headers=HEADERS,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        for index, item in enumerate(manifest, start=1):
            url = item["url"]
            print(
                f"[{index}/{len(manifest)}] 下载：{url}"
            )

            try:
                response = client.get(url)
                response.raise_for_status()

                html = response.text

                if len(html.strip()) < 200:
                    raise ValueError(
                        "HTML 内容过短"
                    )

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

                source = item.get(
                    "source",
                    urlparse(url).netloc,
                )

                record = {
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
                    "source": source,
                    "collected_at": (
                        datetime.now(
                            timezone.utc
                        )
                        .date()
                        .isoformat()
                    ),
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

                records.append(record)

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
                errors.append(
                    {
                        "id": item.get("id"),
                        "url": url,
                        "error_type": type(
                            exc
                        ).__name__,
                        "error": str(exc),
                        **failure,
                    }
                )
                print(
                    f"  下载失败：{type(exc).__name__}"
                )

            time.sleep(1.0)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    error_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            for item in records
        )
        + ("\n" if records else ""),
        encoding="utf-8",
    )

    error_path.write_text(
        "\n".join(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            for item in errors
        )
        + ("\n" if errors else ""),
        encoding="utf-8",
    )

    print(f"成功下载：{len(records)} 条")
    print(f"下载失败：{len(errors)} 条")
    print(f"数据文件：{output_path}")
    print(f"错误文件：{error_path}")


if __name__ == "__main__":
    main()
