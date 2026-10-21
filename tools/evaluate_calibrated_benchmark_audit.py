import json
import os
from collections import Counter
from pathlib import Path

from app.pipeline.calibrated_scoring import (
    calibrated_score_document,
)
from app.pipeline.evaluation import (
    evaluate_documents,
)


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
    "reports/benchmark_audit_calibrated_summary.json"
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

audit_documents = []

for label in labels:
    document_id = label["id"]

    if document_id not in scored_by_id:
        raise ValueError(
            f"评分结果中找不到：{document_id}"
        )

    calibrated = calibrated_score_document(
        scored_by_id[document_id]
    )

    calibrated["decision"] = calibrated[
        "calibrated_decision"
    ]

    calibrated["human_decision"] = label[
        "human_decision"
    ]

    calibrated["audit_action"] = label.get(
        "action",
        "",
    )

    calibrated["audit_reason"] = label.get(
        "reason",
        "",
    )

    audit_documents.append(calibrated)


summary = evaluate_documents(
    audit_documents
)

summary["decision_source"] = (
    "calibrated_scoring_v1"
)

summary["audit_source"] = {
    "scored_file": str(scored_path),
    "labels_file": str(labels_path),
    "labeled_count": len(audit_documents),
}

summary["calibrated_decision_counts"] = dict(
    Counter(
        item["calibrated_decision"]
        for item in audit_documents
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

print(
    f"校准人工审核评估完成："
    f"{summary['total']} 条"
)
print(
    f"校准决策："
    f"{summary['calibrated_decision_counts']}"
)
print(f"准确率：{metrics['accuracy']}")
print(f"精确率：{metrics['precision']}")
print(f"召回率：{metrics['recall']}")
print(f"F1：{metrics['f1']}")
print(
    f"硬误杀数量："
    f"{summary['false_rejection_count']}"
)
print(
    f"误保数量："
    f"{summary['false_acceptance_count']}"
)
print(f"报告文件：{report_path}")
