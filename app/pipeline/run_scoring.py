import json
from collections import Counter
from pathlib import Path

from app.pipeline.scoring import score_documents


input_path = Path(
    "data/processed/near_deduplicated.jsonl"
)

output_path = Path(
    "data/processed/quality_scored.jsonl"
)

summary_path = Path(
    "reports/quality_summary.json"
)


documents = []

for line in input_path.read_text(
    encoding="utf-8"
).splitlines():
    if line.strip():
        documents.append(json.loads(line))


results = score_documents(documents)

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

decision_counts = Counter(
    item["decision"]
    for item in results
)

average_score = sum(
    item["quality_score"]
    for item in results
) / max(len(results), 1)

summary = {
    "total": len(results),
    "decision_counts": dict(decision_counts),
    "average_score": round(
        average_score,
        2,
    ),
}

summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"评分完成：{len(results)} 条")
print(f"决策统计：{dict(decision_counts)}")
print(f"平均分：{summary['average_score']}")
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")