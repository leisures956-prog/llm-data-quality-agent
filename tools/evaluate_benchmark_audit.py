import json
import os
from collections import Counter
from pathlib import Path

from app.pipeline.evaluation import evaluate_documents


scored_path = Path(
    "data/processed/quality_scored.jsonl"
)

labels_path = Path(
    os.environ.get(
        "BENCHMARK_LABELS_PATH",
        "data/labels/benchmark_audit_v1.jsonl",
    ).strip()
)

report_path = Path(
    "reports/benchmark_audit_summary.json"
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


scored_documents = read_jsonl(scored_path)
labels = read_jsonl(labels_path)

scored_by_id = {
    item["id"]: item
    for item in scored_documents
}

label_ids = [item["id"] for item in labels]

if len(label_ids) != len(set(label_ids)):
    raise ValueError("人工标签中存在重复 ID")

missing_ids = [
    item_id
    for item_id in label_ids
    if item_id not in scored_by_id
]

if missing_ids:
    raise ValueError(
        f"评分结果中找不到这些 ID：{missing_ids}"
    )

audit_documents = []

for label in labels:
    item = {
        **scored_by_id[label["id"]],
        "human_decision": label[
            "human_decision"
        ],
        "audit_action": label.get(
            "action",
            "",
        ),
        "audit_reason": label.get(
            "reason",
            "",
        ),
        "manual_false_rejection": label.get(
            "is_false_rejection",
            False,
        ),
    }

    audit_documents.append(item)


summary = evaluate_documents(
    audit_documents
)

summary["audit_source"] = {
    "scored_file": str(scored_path),
    "labels_file": str(labels_path),
    "labeled_count": len(audit_documents),
}

summary["manual_label_counts"] = dict(
    Counter(
        item["human_decision"]
        for item in labels
    )
)

summary["manual_false_rejection_labels"] = sum(
    item.get("is_false_rejection", False)
    for item in labels
)

report_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

metrics = summary["strict_keep_metrics"]

print(f"人工审核评估完成：{summary['total']} 条")
print(
    f"机器决策："
    f"{summary['machine_decision_counts']}"
)
print(
    f"人工决策："
    f"{summary['human_decision_counts']}"
)
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
print(f"报告文件：{report_path}")
