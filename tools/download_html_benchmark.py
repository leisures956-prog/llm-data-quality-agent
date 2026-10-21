import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "data/raw/html_urls_v2.jsonl"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data/raw/html_benchmark_v2.jsonl"
)
ERROR_PATH = (
    PROJECT_ROOT / "data/raw/html_benchmark_v2_errors.jsonl"
)

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


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"输出文件已存在，为避免覆盖：{OUTPUT_PATH}"
        )

    manifest = read_jsonl(MANIFEST_PATH)

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
                errors.append(
                    {
                        "id": item.get("id"),
                        "url": url,
                        "error_type": type(
                            exc
                        ).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"  下载失败：{type(exc).__name__}"
                )

            time.sleep(1.0)

    OUTPUT_PATH.write_text(
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

    ERROR_PATH.write_text(
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
    print(f"数据文件：{OUTPUT_PATH}")
    print(f"错误文件：{ERROR_PATH}")


if __name__ == "__main__":
    main()