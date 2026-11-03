"""Build compact, schema-compatible synthetic scale fixtures.

The fixtures are derived from the frozen raw benchmark, but are explicitly
marked synthetic and never overwrite the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
from pathlib import Path
from typing import Any, Iterator

from app.pipeline.extract import extract_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/html_benchmark_v5_retry_v1.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/performance_fixtures_v1"
DEFAULT_SIZES = (1000, 10000, 100000)
DEFAULT_MAX_TEXT_CHARS = 800


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{resolved_path}")

    rows = []
    for line_number, line in enumerate(
        resolved_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(
                f"JSONL 每行必须是对象：{resolved_path}:{line_number}"
            )
        if not str(item.get("id") or "").strip():
            raise ValueError(
                f"JSONL 缺少有效 id：{resolved_path}:{line_number}"
            )
        rows.append(item)
    return rows


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise ValueError("规模必须是正整数")
        sizes.append(size)
    if not sizes:
        raise ValueError("至少需要一个规模")
    return tuple(dict.fromkeys(sizes))


def _paragraphize(text: str) -> list[str]:
    text = " ".join(str(text or "").split())
    if not text:
        return ["Synthetic performance fixture text."]
    return [
        text[index : index + 240]
        for index in range(0, len(text), 240)
    ] or [text]


def _build_templates(
    base_documents: list[dict[str, Any]],
    max_text_chars: int,
) -> list[dict[str, Any]]:
    templates = []
    for document in base_documents:
        html = str(document.get("html") or "")
        extracted = extract_document(html)
        title = str(extracted.get("title") or document.get("id") or "")
        clean_text = str(extracted.get("clean_text") or "")
        clean_text = clean_text[:max_text_chars]
        paragraphs = _paragraphize(clean_text)
        synthetic_html = (
            "<html><body>"
            f"<h1>{html_lib.escape(title)}</h1>"
            + "".join(
                f"<p>{html_lib.escape(paragraph)}</p>"
                for paragraph in paragraphs
            )
            + "</body></html>"
        )
        templates.append(
            {
                "source_document_id": document["id"],
                "title": title,
                "html": synthetic_html,
                "url": document.get("url", ""),
                "domain": document.get("domain", "unknown"),
                "language": document.get("language", "unknown"),
                "source": document.get("source", ""),
                "collected_at": document.get("collected_at", ""),
                "license": document.get("license", ""),
                "benchmark_category": document.get(
                    "benchmark_category",
                    "synthetic_performance",
                ),
            }
        )

    if not templates:
        raise ValueError("基线数据不能为空")
    return templates


def iter_fixture_documents(
    templates: list[dict[str, Any]],
    size: int,
    mode: str = "repeat",
) -> Iterator[dict[str, Any]]:
    if mode not in {"repeat", "unique"}:
        raise ValueError("mode 只能是 repeat 或 unique")

    for index in range(size):
        template = templates[index % len(templates)]
        document = {
            "id": f"perf_{mode}_{index + 1:06d}",
            "url": template["url"],
            "domain": template["domain"],
            "language": template["language"],
            "source": template["source"],
            "collected_at": template["collected_at"],
            "license": template["license"],
            "benchmark_category": template["benchmark_category"],
            "title": template["title"],
            "html": template["html"],
            "synthetic": True,
            "fixture_mode": mode,
            "fixture_index": index,
            "source_fixture_id": template["source_document_id"],
        }

        if mode == "unique":
            marker = (
                f" Synthetic performance variation {index + 1}."
            )
            marker_html = f"<p>{html_lib.escape(marker)}</p>"
            if "</body>" in document["html"]:
                document["html"] = document["html"].replace(
                    "</body>",
                    f"{marker_html}</body>",
                )
            else:
                document["html"] += marker_html

        yield document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fixture_file(
    path: Path,
    templates: list[dict[str, Any]],
    size: int,
    mode: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in iter_fixture_documents(templates, size, mode):
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
            count += 1

    return {
        "size": count,
        "path": str(path),
        "sha256": _sha256_file(path),
        "synthetic": True,
        "fixture_mode": mode,
    }


def build_scale_fixtures(
    base_documents: list[dict[str, Any]],
    output_dir: str | Path,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    mode: str = "repeat",
) -> dict[str, Any]:
    output_path = resolve_path(output_dir)
    templates = _build_templates(base_documents, max_text_chars)
    entries = []

    for size in sizes:
        fixture_path = output_path / f"fixture_{size:06d}.jsonl"
        entry = write_fixture_file(
            fixture_path,
            templates,
            size,
            mode,
        )
        entry["path"] = str(fixture_path)
        entries.append(entry)

    manifest = {
        "fixture_version": "performance_fixtures_v1",
        "base_document_count": len(base_documents),
        "max_text_chars": max_text_chars,
        "fixture_mode": mode,
        "synthetic": True,
        "fixtures": entries,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成性能与规模验证用的合成 JSONL 数据"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in DEFAULT_SIZES),
        help="逗号分隔的规模，例如 1000,10000,100000",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
    )
    parser.add_argument(
        "--mode",
        choices=("repeat", "unique"),
        default="repeat",
        help="repeat 用于重复数据压力；unique 用于近似全唯一压力",
    )
    args = parser.parse_args()

    if args.max_text_chars <= 0:
        raise ValueError("--max-text-chars 必须是正整数")

    manifest = build_scale_fixtures(
        load_jsonl(args.input),
        args.output_dir,
        sizes=parse_sizes(args.sizes),
        max_text_chars=args.max_text_chars,
        mode=args.mode,
    )
    print(
        f"性能样本生成完成：{len(manifest['fixtures'])} 个规模"
    )
    for fixture in manifest["fixtures"]:
        print(
            f"{fixture['size']} 条：{fixture['path']}"
        )
    print(
        "清单文件："
        f"{resolve_path(args.output_dir) / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
