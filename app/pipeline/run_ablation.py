import json
from pathlib import Path

from app.pipeline.ablation import (
    build_rule_variants,
    evaluate_variant,
)


input_path = Path(
    "data/processed/repaired.jsonl"
)

summary_path = Path(
    "reports/ablation_summary.json"
)

detail_path = Path(
    "reports/ablation_document_results.jsonl"
)


documents = [
    json.loads(line)
    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

variants = build_rule_variants()

summaries = []
detail_records = []

for variant_name, rules in variants.items():
    summary, results = evaluate_variant(
        documents,
        variant_name,
        rules,
    )

    summaries.append(summary)

    for item in results:
        detail_records.append(
            {
                "variant_name": variant_name,
                "rule_version": rules[
                    "version"
                ],
                "id": item.get("id"),
                "quality_score": item.get(
                    "calibrated_quality_score"
                ),
                "decision": item.get(
                    "calibrated_decision"
                ),
                "human_decision": item.get(
                    "human_decision"
                ),
                "is_near_duplicate": item.get(
                    "is_near_duplicate",
                    False,
                ),
            }
        )

baseline_counts = summaries[0][
    "decision_counts"
]

for summary in summaries:
    current_counts = summary[
        "decision_counts"
    ]

    summary["delta_vs_baseline"] = {
        "keep": current_counts.get(
            "keep",
            0,
        )
        - baseline_counts.get(
            "keep",
            0,
        ),
        "review": current_counts.get(
            "review",
            0,
        )
        - baseline_counts.get(
            "review",
            0,
        ),
        "reject": current_counts.get(
            "reject",
            0,
        )
        - baseline_counts.get(
            "reject",
            0,
        ),
    }


summary_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary_path.write_text(
    json.dumps(
        {
            "input": str(input_path),
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
        for item in detail_records
    )
    + "\n",
    encoding="utf-8",
)

print(
    f"消融实验完成：{len(summaries)} 个策略"
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