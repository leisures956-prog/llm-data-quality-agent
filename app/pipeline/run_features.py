import json
import os
from pathlib import Path

from app.pipeline.extract import extract_document
from app.pipeline.features import compute_features
from app.pipeline.dataset_paths import (
    get_dataset_path,
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

output_path = Path("data/processed/features.jsonl")

results = []

for line in input_path.read_text(
    encoding="utf-8"
).splitlines():
    if not line.strip():
        continue

    raw_document = json.loads(line)

    html = raw_document.get("html", "")

    extracted = extract_document(html)

    features = compute_features(
        html=html,
        title=extracted["title"],
        clean_text=extracted["clean_text"],
    )

    metadata_missing_fields = [
        field
        for field in (
            "source",
            "collected_at",
            "license",
        )
        if not str(
            raw_document.get(field) or ""
        ).strip()
    ]

    results.append({
        "id": raw_document["id"],
        "url": raw_document.get("url", ""),
        "domain": raw_document.get(
            "domain",
            "unknown",
        ),
        "language": raw_document.get(
            "language",
            "unknown",
        ),
        "source": raw_document.get(
            "source",
            "",
        ),
        "collected_at": raw_document.get(
            "collected_at",
            "",
        ),
        "license": raw_document.get(
            "license",
            "",
        ),
        "benchmark_category": raw_document.get(
            "benchmark_category",
            "",
        ),
        "dataset_version": dataset_version,
        "pipeline_version": pipeline_version,
        "run_id": run_id,
        "metadata_complete": not metadata_missing_fields,
        "metadata_missing_fields": (
            metadata_missing_fields
        ),
        "title": extracted["title"],
        "clean_text": extracted["clean_text"],
        **features,
    })

output_path.write_text(
    "\n".join(
        json.dumps(
            item,
            ensure_ascii=False,
        )
        for item in results
    ),
    encoding="utf-8",
)

print(f"特征计算完成：{len(results)} 条")
print(f"输出文件：{output_path}")
