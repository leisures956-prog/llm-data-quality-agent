from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tools.download_failure import classify_error_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data/raw/html_urls_v3.jsonl"
)
DEFAULT_ERRORS_PATH = (
    PROJECT_ROOT
    / "data/raw/html_benchmark_v3_retry_v1_errors.jsonl"
)
DEFAULT_REPLACEMENTS_PATH = (
    PROJECT_ROOT / "data/raw/source_replacements_v4.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data/raw/html_urls_v4.jsonl"
)
DEFAULT_PLAN_PATH = (
    PROJECT_ROOT
    / "reports/html_benchmark_v4/source_replacement_plan.json"
)
DEFAULT_TEMPLATE_PLAN_PATH = (
    PROJECT_ROOT
    / "reports/html_benchmark_v4/source_replacement_template_plan.json"
)


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


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "").strip()


def _source(item: dict[str, Any]) -> str:
    source = item.get("source") or item.get("source_key")

    if source:
        return str(source)

    return "unknown"


def _category(item: dict[str, Any]) -> str:
    return str(
        item.get("benchmark_category")
        or ""
    )


def _validate_unique_ids(
    rows: list[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    result = {}

    for item in rows:
        item_id = _item_id(item)

        if not item_id:
            raise ValueError(
                f"{label} 存在空 ID"
            )

        if item_id in result:
            raise ValueError(
                f"{label} 存在重复 ID：{item_id}"
            )

        result[item_id] = item

    return result


def _valid_http_url(value: object) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(
        parsed.netloc
    )


def select_blocked_failures(
    manifest_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select source-level blocked failures for replacement.

    Transient network failures are intentionally excluded because they belong
    to the retry stage. Unknown error IDs are returned separately so the
    replacement plan cannot silently lose provenance.
    """

    manifest_map = _validate_unique_ids(
        manifest_rows,
        "原始清单",
    )
    selected = []
    selected_ids = set()
    unknown_ids = []

    for error in error_rows:
        item_id = _item_id(error)
        failure = classify_error_record(error)

        if failure["error_class"] != (
            "blocked_or_unavailable"
        ):
            continue

        if item_id not in manifest_map:
            unknown_ids.append(item_id)
            continue

        if item_id in selected_ids:
            raise ValueError(
                f"错误文件中存在重复阻断 ID：{item_id}"
            )

        selected.append(
            {
                "id": item_id,
                "manifest": manifest_map[item_id],
                "error": error,
                "failure": failure,
            }
        )
        selected_ids.add(item_id)

    return selected, sorted(
        set(item_id for item_id in unknown_ids if item_id)
    )


def build_replacement_template(
    blocked_failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    template = []

    for failure in blocked_failures:
        original = failure["manifest"]
        template.append(
            {
                "replacement_for": failure["id"],
                "replacement_id": "",
                "url": "",
                "title": "",
                "source": "",
                "language": original.get(
                    "language",
                    "unknown",
                ),
                "domain": original.get(
                    "domain",
                    "unknown",
                ),
                "benchmark_category": _category(
                    original
                ),
                "license": "page_terms_to_verify",
                "original_url": original.get(
                    "url",
                    "",
                ),
                "original_source": _source(original),
                "failure_class": failure[
                    "failure"
                ]["error_class"],
                "notes": "请填写允许访问且内容主题相近的替代网页",
            }
        )

    return template


def _generated_replacement_id(
    original_id: str,
) -> str:
    suffix = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        original_id,
    ).strip("_")
    return f"html_v4_replacement_{suffix}"


def _replacement_record(
    original: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    original_id = _item_id(original)
    replacement_url = str(
        replacement.get("url") or ""
    ).strip()

    if not _valid_http_url(replacement_url):
        raise ValueError(
            "替代记录必须提供有效的 http/https URL："
            f"{original_id}"
        )

    replacement_id = str(
        replacement.get("replacement_id")
        or replacement.get("id")
        or _generated_replacement_id(original_id)
    ).strip()

    if not replacement_id:
        raise ValueError(
            f"替代记录生成了空 ID：{original_id}"
        )

    source = str(
        replacement.get("source")
        or urlparse(replacement_url).netloc
    ).strip()

    if not source:
        raise ValueError(
            f"替代记录缺少 source：{original_id}"
        )

    language = str(
        replacement.get(
            "language",
            original.get("language", "unknown"),
        )
    ).strip()
    domain = str(
        replacement.get(
            "domain",
            original.get("domain", "unknown"),
        )
    ).strip()
    category = str(
        replacement.get(
            "benchmark_category",
            _category(original),
        )
    ).strip()
    license_value = str(
        replacement.get(
            "license",
            "page_terms_to_verify",
        )
    ).strip()

    return {
        "id": replacement_id,
        "url": replacement_url,
        "domain": domain or "unknown",
        "language": language or "unknown",
        "source": source,
        "license": license_value,
        "benchmark_category": category,
        "title": str(
            replacement.get("title") or ""
        ).strip(),
        "replaced_from": original_id,
        "original_url": original.get(
            "url",
            "",
        ),
        "replacement_reason": (
            "source_blocked_or_unavailable"
        ),
    }


def build_replacement_manifest(
    manifest_rows: list[dict[str, Any]],
    blocked_failures: list[dict[str, Any]],
    replacement_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_map = _validate_unique_ids(
        manifest_rows,
        "原始清单",
    )
    replacement_map = {}

    for replacement in replacement_rows:
        original_id = str(
            replacement.get("replacement_for") or ""
        ).strip()

        if not original_id:
            raise ValueError(
                "替代清单缺少 replacement_for"
            )

        if original_id in replacement_map:
            raise ValueError(
                "替代清单存在重复 replacement_for："
                f"{original_id}"
            )

        replacement_map[original_id] = replacement

    blocked_ids = {
        failure["id"]
        for failure in blocked_failures
    }
    missing_ids = sorted(
        blocked_ids - set(replacement_map)
    )
    unknown_replacement_ids = sorted(
        set(replacement_map) - blocked_ids
    )

    if missing_ids:
        raise ValueError(
            "以下阻断 ID 尚未提供替代来源："
            f"{missing_ids}"
        )

    if unknown_replacement_ids:
        raise ValueError(
            "替代清单包含当前阻断列表之外的 ID："
            f"{unknown_replacement_ids}"
        )

    output_rows = []
    output_ids = set()
    output_urls = set()
    replacements = []

    for original in manifest_rows:
        original_id = _item_id(original)

        if original_id not in blocked_ids:
            record = dict(original)
        else:
            record = _replacement_record(
                original,
                replacement_map[original_id],
            )
            replacements.append(
                {
                    "original_id": original_id,
                    "replacement_id": record["id"],
                    "original_url": original.get(
                        "url",
                        "",
                    ),
                    "replacement_url": record["url"],
                    "original_source": _source(
                        original
                    ),
                    "replacement_source": record[
                        "source"
                    ],
                    "language": record["language"],
                    "benchmark_category": record[
                        "benchmark_category"
                    ],
                }
            )

        if record["id"] in output_ids:
            raise ValueError(
                f"生成结果存在重复 ID：{record['id']}"
            )

        if record["url"] in output_urls:
            raise ValueError(
                f"生成结果存在重复 URL：{record['url']}"
            )

        output_ids.add(record["id"])
        output_urls.add(record["url"])
        output_rows.append(record)

    plan = {
        "plan_version": "source_replacement_plan_v1",
        "status": "ready",
        "original_manifest_count": len(manifest_rows),
        "blocked_count": len(blocked_ids),
        "replacement_count": len(replacements),
        "output_manifest_count": len(output_rows),
        "replacement_source_counts": dict(
            sorted(
                Counter(
                    item["replacement_source"]
                    for item in replacements
                ).items()
            )
        ),
        "replacement_language_counts": dict(
            sorted(
                Counter(
                    item["language"]
                    for item in replacements
                ).items()
            )
        ),
        "replacements": replacements,
    }

    return output_rows, plan


def _write_json(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    resolved_path = resolve_path(path)
    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    resolved_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "生成来源替换模板，或批量构建新的 HTML URL 清单"
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="原始 URL 清单 JSONL 路径",
    )
    parser.add_argument(
        "--errors",
        default=str(DEFAULT_ERRORS_PATH),
        help="下载错误 JSONL 路径",
    )
    parser.add_argument(
        "--template-output",
        default="",
        help="只生成替换模板，不构建 v4 清单",
    )
    parser.add_argument(
        "--replacements",
        default=str(DEFAULT_REPLACEMENTS_PATH),
        help="已填写的替代来源 JSONL 路径",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="新 URL 清单输出路径",
    )
    parser.add_argument(
        "--plan-output",
        default=str(DEFAULT_PLAN_PATH),
        help="替换计划报告路径",
    )

    args = parser.parse_args()
    manifest_path = resolve_path(args.manifest)
    errors_path = resolve_path(args.errors)
    manifest_rows = load_jsonl(manifest_path)
    error_rows = load_jsonl(errors_path)
    blocked_failures, unknown_error_ids = (
        select_blocked_failures(
            manifest_rows,
            error_rows,
        )
    )

    if args.template_output:
        template_path = resolve_path(args.template_output)

        if template_path.exists():
            raise FileExistsError(
                f"模板文件已存在，为避免覆盖：{template_path}"
            )

        template_rows = build_replacement_template(
            blocked_failures
        )
        write_jsonl(template_path, template_rows)
        template_plan = {
            "plan_version": (
                "source_replacement_template_plan_v1"
            ),
            "status": "template_required",
            "manifest_path": str(manifest_path),
            "errors_path": str(errors_path),
            "blocked_count": len(blocked_failures),
            "template_count": len(template_rows),
            "blocked_ids": [
                failure["id"]
                for failure in blocked_failures
            ],
            "unknown_error_ids": unknown_error_ids,
            "next_step": (
                "填写 template 中的 url/source/title 等字段，"
                "再作为 --replacements 输入"
            ),
        }
        plan_path = resolve_path(
            DEFAULT_TEMPLATE_PLAN_PATH
        )

        if plan_path.exists():
            raise FileExistsError(
                "模板计划已存在，为避免覆盖："
                f"{plan_path}"
            )

        _write_json(plan_path, template_plan)
        print(
            f"替换模板生成：{len(template_rows)} 条"
        )
        print(f"模板文件：{template_path}")
        print(f"计划文件：{plan_path}")
        return

    replacements_path = resolve_path(args.replacements)
    output_path = resolve_path(args.output)
    plan_path = resolve_path(args.plan_output)

    if not replacements_path.exists():
        raise FileNotFoundError(
            "替代来源清单不存在，请先使用 --template-output 生成模板："
            f"{replacements_path}"
        )

    if output_path.exists():
        raise FileExistsError(
            f"v4 清单已存在，为避免覆盖：{output_path}"
        )

    if plan_path.exists():
        raise FileExistsError(
            f"替换计划已存在，为避免覆盖：{plan_path}"
        )

    replacement_rows = load_jsonl(replacements_path)
    output_rows, plan = build_replacement_manifest(
        manifest_rows,
        blocked_failures,
        replacement_rows,
    )
    plan.update(
        {
            "manifest_path": str(manifest_path),
            "errors_path": str(errors_path),
            "replacements_path": str(
                replacements_path
            ),
            "unknown_error_ids": unknown_error_ids,
        }
    )
    write_jsonl(output_path, output_rows)
    _write_json(plan_path, plan)

    print(
        f"来源替换清单生成：{len(output_rows)} 条"
    )
    print(
        f"替换来源：{len(plan['replacements'])} 条"
    )
    print(f"新清单：{output_path}")
    print(f"替换计划：{plan_path}")


if __name__ == "__main__":
    main()
