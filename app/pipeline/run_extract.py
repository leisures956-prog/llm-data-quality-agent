import json
from pathlib import Path

from app.pipeline.extract import extract_document
from app.pipeline.dataset_paths import (
    get_dataset_path,
)

input_path = get_dataset_path()
output_path = Path("data/processed/cleaned.jsonl")

results = []

for line in input_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    item = json.loads(line)
    result = extract_document(item["html"])

    results.append({
        "id": item["id"],
        "url": item["url"],
        **result,
    })

output_path.write_text(
    "\n".join(json.dumps(item, ensure_ascii=False) for item in results),
    encoding="utf-8",
)

print(f"处理完成：{len(results)} 条")
print(f"输出文件：{output_path}")