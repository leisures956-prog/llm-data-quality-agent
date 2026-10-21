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
    "data/processed/repaired.jsonl"
)

output_path = Path(
    "data/processed/calibrated_scored.jsonl"
)

summary_path = Path(
    "reports/calibrated_summary.json"
)


documents = [
    json.loads(line)
    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

results = calibrated_score_documents(documents)

false_rejection_count = sum(
    item.get("calibrated_decision") == "reject"
    and item.get("human_decision") == "keep"
    for item in results
)

false_acceptance_count = sum(
    item.get("calibrated_decision") == "keep"
    and item.get("human_decision") == "reject"
    for item in results
)

review_keep_count = sum(
    item.get("calibrated_decision") == "review"
    and item.get("human_decision") == "keep"
    for item in results
)

review_count = sum(
    item.get("calibrated_decision") == "review"
    for item in results
)
review_rate = round(
    review_count / max(len(results), 1),
    4,
)
review_budget_rate = float(
    load_quality_rules()
    .get("review_budget", {})
    .get("max_rate", 1.0)
)

summary = {
    "total": len(results),
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
    "false_rejection_count": (
        false_rejection_count
    ),
    "false_acceptance_count": (
        false_acceptance_count
    ),
    "review_keep_count": review_keep_count,
    "review_rate": review_rate,
    "review_budget_rate": review_budget_rate,
    "review_budget_exceeded": (
        review_rate > review_budget_rate
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

print(f"校准评分完成：{len(results)} 条")
print(
    f"校准决策："
    f"{summary['calibrated_decision_counts']}"
)
print(
    f"误杀数量："
    f"{summary['false_rejection_count']}"
)
print(
    f"误保数量："
    f"{summary['false_acceptance_count']}"
)
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")
