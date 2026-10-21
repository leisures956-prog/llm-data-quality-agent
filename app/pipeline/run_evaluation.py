import json
from pathlib import Path

from app.pipeline.evaluation import (
    evaluate_documents,
)


input_path = Path(
    "data/processed/classified.jsonl"
)

output_path = Path(
    "reports/evaluation_summary.json"
)


documents = [
    json.loads(line)
    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

summary = evaluate_documents(documents)

output_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

metrics = summary["strict_keep_metrics"]

print(f"评估完成：{summary['total']} 条")
print(f"准确率：{metrics['accuracy']}")
print(f"精确率：{metrics['precision']}")
print(f"召回率：{metrics['recall']}")
print(f"F1：{metrics['f1']}")
print(
    f"误杀数量："
    f"{summary['false_rejection_count']}"
)
print(
    f"误保数量："
    f"{summary['false_acceptance_count']}"
)
print(
    f"待复核数量："
    f"{summary['review_count']}"
)
print(f"报告文件：{output_path}")