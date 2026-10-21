import json
from collections import Counter, defaultdict
from pathlib import Path

from app.pipeline.classification import (
    classify_document,
)


input_path = Path(
    "data/processed/"
    "production_decisions_benchmark_v1.jsonl"
)

output_path = Path(
    "data/processed/"
    "classified_benchmark_v1.jsonl"
)

summary_path = Path(
    "reports/"
    "distribution_benchmark_summary.json"
)


documents = [
    json.loads(line)
    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

results = [
    classify_document(document)
    for document in documents
]

language_topic_counts = defaultdict(Counter)

for item in results:
    language_topic_counts[
        item["language"]
    ][item["topic"]] += 1

summary = {
    "total": len(results),
    "language_counts": dict(
        Counter(
            item["language"]
            for item in results
        )
    ),
    "detected_language_counts": dict(
        Counter(
            item["detected_language"]
            for item in results
        )
    ),
    "topic_counts": dict(
        Counter(
            item["topic"]
            for item in results
        )
    ),
    "quality_counts": dict(
        Counter(
            item["quality_bucket"]
            for item in results
        )
    ),
    "length_bucket_counts": dict(
        Counter(
            item["length_bucket"]
            for item in results
        )
    ),
    "language_topic_counts": {
        language: dict(topic_counts)
        for language, topic_counts
        in language_topic_counts.items()
    },
}

output_path.write_text(
    "\n".join(
        json.dumps(
            item,
            ensure_ascii=False,
        )
        for item in results
    )
    + "\n",
    encoding="utf-8",
)

summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(f"分类完成：{len(results)} 条")
print(
    f"语言分布："
    f"{summary['language_counts']}"
)
print(
    f"主题分布："
    f"{summary['topic_counts']}"
)
print(
    f"质量分布："
    f"{summary['quality_counts']}"
)
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")