import json
import os
from pathlib import Path

from app.pipeline.dataset_registry import (
    summarize_dataset,
    validate_raw_document,
)
from app.pipeline.dataset_paths import (
    get_dataset_path,
)
from app.pipeline.source_registry import (
    load_source_registry,
)

input_path = get_dataset_path()

dataset_version = os.environ.get(
    "DATASET_VERSION",
    input_path.stem,
)

pipeline_version = os.environ.get(
    "PIPELINE_VERSION",
    "quality_pipeline_v1",
)

run_id = os.environ.get(
    "RUN_ID",
    dataset_version,
)

report_path = Path(
    os.environ.get(
        "DATASET_REPORT_PATH",
        "reports/dataset_manifest.json",
    )
)

source_registry_path = Path(
    os.environ.get(
        "SOURCE_REGISTRY_PATH",
        "config/source_registry.json",
    )
)

source_registry = load_source_registry(
    source_registry_path
)

documents = []
errors = []
valid_count = 0

for line_number, line in enumerate(
    input_path.read_text(
        encoding="utf-8"
    ).splitlines(),
    start=1,
):
    if not line.strip():
        continue

    try:
        document = json.loads(line)
    except json.JSONDecodeError as exc:
        errors.append(
            {
                "line": line_number,
                "errors": [
                    f"JSON 格式错误：{exc.msg}"
                ],
            }
        )
        continue

    documents.append(document)

    document_errors = validate_raw_document(
        document,
        line_number,
    )

    if document_errors:
        errors.append(
            {
                "line": line_number,
                "id": document.get("id"),
                "errors": document_errors,
            }
        )
    else:
        valid_count += 1

summary = summarize_dataset(
    documents,
    source_registry=source_registry,
)
summary["valid_count"] = valid_count
summary["invalid_count"] = len(
    documents
) - valid_count

payload = {
    "schema_version": "raw_web_v1",
    "dataset_path": str(input_path),
    "provenance": {
        "dataset_version": dataset_version,
        "pipeline_version": pipeline_version,
        "run_id": run_id,
    },
    "source_registry_path": str(
        source_registry_path
    ),
    "summary": summary,
    "errors": errors,
}

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.write_text(
    json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(f"数据集检查完成：{len(documents)} 条")
print(f"有效数据：{valid_count} 条")
print(
    f"问题数据："
    f"{summary['invalid_count']} 条"
)
print(f"报告文件：{report_path}")

if errors:
    raise SystemExit(1)
