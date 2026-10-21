import json
from pathlib import Path

from app.pipeline.deduplicate import (
    annotate_exact_duplicates,
    select_unique_documents,
)


input_path = Path(
    "data/processed/features.jsonl"
)

annotated_output_path = Path(
    "data/processed/deduplicated.jsonl"
)

unique_output_path = Path(
    "data/processed/unique.jsonl"
)


documents = []

for line in input_path.read_text(
    encoding="utf-8"
).splitlines():
    if line.strip():
        documents.append(json.loads(line))


annotated_documents = annotate_exact_duplicates(
    documents
)

unique_documents = select_unique_documents(
    annotated_documents
)


def write_jsonl(
    path: Path,
    documents: list[dict],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            for item in documents
        ),
        encoding="utf-8",
    )


write_jsonl(
    annotated_output_path,
    annotated_documents,
)

write_jsonl(
    unique_output_path,
    unique_documents,
)


duplicate_documents = sum(
    item["is_duplicate"]
    for item in annotated_documents
)

duplicate_groups = {
    item["duplicate_group"]
    for item in annotated_documents
    if item["duplicate_group"] is not None
}

print(f"原始文档：{len(documents)} 条")
print(f"重复文档：{duplicate_documents} 条")
print(f"重复分组：{len(duplicate_groups)} 组")
print(f"去重后保留：{len(unique_documents)} 条")
print(
    f"审计文件：{annotated_output_path}"
)
print(
    f"唯一数据：{unique_output_path}"
)