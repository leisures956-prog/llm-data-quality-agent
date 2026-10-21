import json
from pathlib import Path

from app.pipeline.near_deduplicate import (
    annotate_near_duplicates,
)


input_path = Path(
    "data/processed/unique.jsonl"
)

output_path = Path(
    "data/processed/near_deduplicated.jsonl"
)

report_path = Path(
    "reports/near_duplicate_pairs.jsonl"
)


documents = []

for line in input_path.read_text(
    encoding="utf-8"
).splitlines():
    if line.strip():
        documents.append(json.loads(line))


results, pairs = annotate_near_duplicates(
    documents,
    threshold=0.75,
)


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

report_path.write_text(
    "\n".join(
        json.dumps(
            pair,
            ensure_ascii=False,
        )
        for pair in pairs
    ),
    encoding="utf-8",
)

near_duplicate_documents = sum(
    item["is_near_duplicate"]
    for item in results
)

near_duplicate_groups = {
    item["near_duplicate_group"]
    for item in results
    if item["near_duplicate_group"] is not None
}

print(f"输入文档：{len(documents)} 条")
print(
    f"近似重复文档：{near_duplicate_documents} 条"
)
print(
    f"近似重复分组：{len(near_duplicate_groups)} 组"
)
print(f"结果文件：{output_path}")
print(f"相似度报告：{report_path}")