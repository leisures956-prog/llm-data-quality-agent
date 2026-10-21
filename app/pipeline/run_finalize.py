import json
from collections import Counter
from pathlib import Path

from app.pipeline.decision_policy import (
    apply_production_decision,
)


input_path = Path(
    "data/processed/calibrated_scored.jsonl"
)

output_path = Path(
    "data/processed/production_decisions.jsonl"
)

summary_path = Path(
    "reports/decision_source_summary.json"
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
        "calibrated_scoring_v1"
    ),
    "production_decision_counts": dict(
        Counter(
            item["production_decision"]
            for item in results
        )
    ),
    "human_final_decision_counts": dict(
        Counter(
            item.get(
                "final_decision",
                "unknown",
            )
            for item in results
        )
    ),
    "changed_from_original_machine": sum(
        item["production_decision"]
        != item.get("decision")
        for item in results
    ),
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
    f"与原始机器决策不同："
    f"{summary['changed_from_original_machine']} 条"
)
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")