import json
from collections import Counter
from pathlib import Path

from app.pipeline.decision_policy import (
    apply_production_decision,
    PRODUCTION_DECISION_SOURCE,
)


input_path = Path(
    "data/processed/"
    "calibrated_scored_benchmark_v1.jsonl"
)

output_path = Path(
    "data/processed/"
    "production_decisions_benchmark_v1.jsonl"
)

summary_path = Path(
    "reports/"
    "decision_source_benchmark_summary.json"
)


documents = [
    json.loads(line)
    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

results = [
    apply_production_decision(document)
    for document in documents
]

summary = {
    "total": len(results),
    "production_decision_source": (
        PRODUCTION_DECISION_SOURCE
    ),
    "production_decision_counts": dict(
        Counter(
            item["production_decision"]
            for item in results
        )
    ),
    "review_ids": [
        item["id"]
        for item in results
        if item["production_decision"]
        == "review"
    ],
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

print(f"正式决策生成：{len(results)} 条")
print(
    f"正式决策："
    f"{summary['production_decision_counts']}"
)
print(
    f"生产决策来源："
    f"{summary['production_decision_source']}"
)
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")