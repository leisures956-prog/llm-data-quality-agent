import json
from collections import Counter
from pathlib import Path

from app.pipeline.calibrated_scoring import (
    calibrated_score_documents,
)
from app.pipeline.rule_config import (
    load_quality_rules,
)


input_path = Path(
    "data/processed/quality_scored.jsonl"
)

output_path = Path(
    "data/processed/"
    "calibrated_scored_benchmark_v1.jsonl"
)

summary_path = Path(
    "reports/calibrated_benchmark_summary.json"
)


documents = [
    json.loads(line)
    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

results = calibrated_score_documents(documents)

summary = {
    "total": len(results),
    "rule_version": (
        results[0].get(
            "quality_rule_version"
        )
        if results
        else None
    ),
    "content_decision_counts": dict(
        Counter(
            item["content_decision"]
            for item in results
        )
    ),
    "calibrated_decision_counts": dict(
        Counter(
            item["calibrated_decision"]
            for item in results
        )
    ),
    "duplicate_status_counts": dict(
        Counter(
            item["duplicate_status"]
            for item in results
        )
    ),
    "review_ids": [
        item["id"]
        for item in results
        if item["calibrated_decision"]
        == "review"
    ],
}
summary["review_rate"] = round(
    len(summary["review_ids"])
    / max(summary["total"], 1),
    4,
)
summary["review_budget_rate"] = float(
    load_quality_rules()
    .get("review_budget", {})
    .get("max_rate", 1.0)
)
summary["review_budget_exceeded"] = (
    summary["review_rate"]
    > summary["review_budget_rate"]
)

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

print(f"校准评分完成：{len(results)} 条")
print(
    f"校准决策："
    f"{summary['calibrated_decision_counts']}"
)
print(
    f"人工复核数量："
    f"{len(summary['review_ids'])}"
)
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")
