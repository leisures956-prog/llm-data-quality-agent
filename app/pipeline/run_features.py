import json
import os
from pathlib import Path

from app.pipeline.extract import extract_document
from app.pipeline.features import compute_features
from app.pipeline.dataset_paths import (
    get_dataset_path,
)
from app.pipeline.source_registry import (
    annotate_document_source,
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

source_registry_path = Path(
    os.environ.get(
        "SOURCE_REGISTRY_PATH",
        "config/source_registry.json",
    )
)

source_registry = load_source_registry(
    source_registry_path
)

output_path = Path("data/processed/features.jsonl")

results = []

for line in input_path.read_text(
    encoding="utf-8"
).splitlines():
    if not line.strip():
        continue

    raw_document = json.loads(line)

    source_info = annotate_document_source(
        raw_document,
        source_registry,
    )

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
        "source_registry_version": source_info[
            "source_registry_version"
        ],
        "source_key": source_info[
            "source_key"
        ],
        "source_resolution": source_info[
            "source_resolution"
        ],
        "source_registered": source_info[
            "source_registered"
        ],
        "source_registry_status": source_info[
            "source_registry_status"
        ],
        "source_registry_name": source_info[
            "source_registry_name"
        ],
        "license_status": source_info[
            "license_status"
        ],
        "usage_policy": source_info[
            "usage_policy"
        ],
        "experiment_eligible": source_info[
            "experiment_eligible"
        ],
        "release_eligible": source_info[
            "release_eligible"
        ],
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
