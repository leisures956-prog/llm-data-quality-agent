import json
import os
from pathlib import Path

from app.pipeline.ablation import (
    build_rule_variants,
    evaluate_variant,
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

summary_path = Path(
    "reports/"
    "ablation_benchmark_audit_summary.json"
)

detail_path = Path(
    "reports/"
    "ablation_benchmark_audit_details.jsonl"
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
    document = {
        **scored_by_id[label["id"]],
        "human_decision": label[
            "human_decision"
        ],
    }
    audit_documents.append(document)


variants = build_rule_variants()

summaries = []
details = []

for variant_name, rules in variants.items():
    summary, results = evaluate_variant(
        audit_documents,
        variant_name,
        rules,
    )

    summaries.append(summary)

    for item in results:
        details.append(
            {
                "variant_name": variant_name,
                "rule_version": rules[
                    "version"
                ],
                "id": item["id"],
                "calibrated_quality_score": item[
                    "calibrated_quality_score"
                ],
                "calibrated_decision": item[
                    "calibrated_decision"
                ],
                "human_decision": item[
                    "human_decision"
                ],
            }
        )


baseline = summaries[0]["decision_counts"]

for summary in summaries:
    current = summary["decision_counts"]

    summary["delta_vs_baseline"] = {
        "keep": current.get("keep", 0)
        - baseline.get("keep", 0),
        "review": current.get("review", 0)
        - baseline.get("review", 0),
        "reject": current.get("reject", 0)
        - baseline.get("reject", 0),
    }


summary_path.write_text(
    json.dumps(
        {
            "input": {
                "scored_file": str(
                    scored_path
                ),
                "labels_file": str(
                    labels_path
                ),
                "labeled_count": len(
                    audit_documents
                ),
            },
            "variant_count": len(summaries),
            "variants": summaries,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

detail_path.write_text(
    "\n".join(
        json.dumps(
            item,
            ensure_ascii=False,
        )
        for item in details
    )
    + "\n",
    encoding="utf-8",
)

print(
    f"消融实验完成："
    f"{len(summaries)} 个策略"
)

for summary in summaries:
    print(
        summary["variant_name"],
        summary["decision_counts"],
        "误杀：",
        summary["false_rejection_count"],
        "误保：",
        summary["false_acceptance_count"],
    )

print(f"汇总报告：{summary_path}")
print(f"明细报告：{detail_path}")
