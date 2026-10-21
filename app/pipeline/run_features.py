import json
from pathlib import Path

from app.pipeline.extract import extract_document
from app.pipeline.features import compute_features
from app.pipeline.dataset_paths import (
    get_dataset_path,
)

input_path = get_dataset_path()
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