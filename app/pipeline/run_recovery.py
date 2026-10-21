import json
from pathlib import Path

from app.pipeline.recovery import (
    apply_manual_labels,
    summarize_recovery,
)


input_path = Path("data/processed/quality_scored.jsonl")
labels_path = Path("data/labels/manual_labels.jsonl")
output_path = Path("data/processed/recovered.jsonl")
summary_path = Path("reports/recovery_summary.json")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


documents = read_jsonl(input_path)
labels = read_jsonl(labels_path)
results = apply_manual_labels(documents, labels)
summary = summarize_recovery(results)

output_path.write_text(
    "\n".join(
        json.dumps(item, ensure_ascii=False)
        for item in results
    )
    + "\n",
    encoding="utf-8",
)

summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2)
    + "\n",
    encoding="utf-8",
)

print(f"人工标注：{len(labels)} 条")
print(f"误杀样本：{summary['false_rejection_count']} 条")
print(f"成功恢复：{summary['recovered_count']} 条")
print(f"恢复率：{summary['recovery_rate']}%")
print(f"最终决策：{summary['final_decision_counts']}")
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")
