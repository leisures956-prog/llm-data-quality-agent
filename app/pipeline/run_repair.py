import json
from collections import Counter
from pathlib import Path

from app.pipeline.features import compute_features
from app.pipeline.repair import repair_document
from app.pipeline.scoring import score_document


input_path = Path("data/processed/recovered.jsonl")
raw_input_path = Path("data/raw/sample.jsonl")
output_path = Path("data/processed/repaired.jsonl")
summary_path = Path("reports/repair_summary.json")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


documents = read_jsonl(input_path)
raw_documents = read_jsonl(raw_input_path)
raw_by_id = {
    document["id"]: document
    for document in raw_documents
}

results = []

for document in documents:
    document_id = document["id"]
    raw_document = raw_by_id.get(document_id)
    if raw_document is None:
        raise ValueError(
            f"找不到文档对应的原始 HTML：{document_id}"
        )

    repaired = repair_document(document)
    html = raw_document.get("html", "")

    repaired_features = compute_features(
        html=html,
        title=document.get("title", ""),
        clean_text=repaired["clean_text"],
    )

    candidate = {
        **repaired,
        **repaired_features,
    }
    post_repair_score = score_document(candidate)

    before_score = document.get("quality_score", 0)
    before_decision = document.get("decision", "review")

    results.append(
        {
            **candidate,
            "post_repair_quality_score": post_repair_score[
                "quality_score"
            ],
            "post_repair_decision": post_repair_score[
                "decision"
            ],
            "post_repair_reasons": post_repair_score[
                "reasons"
            ],
            "score_delta": post_repair_score[
                "quality_score"
            ] - before_score,
            "decision_changed": (
                before_decision
                != post_repair_score["decision"]
            ),
        }
    )

action_counts = Counter(
    action
    for item in results
    for action in item["repair_actions"]
)

summary = {
    "total": len(results),
    "repaired_document_count": sum(
        item["repair_applied"]
        for item in results
    ),
    "score_improved_count": sum(
        item["score_delta"] > 0
        for item in results
    ),
    "decision_changed_count": sum(
        item["decision_changed"]
        for item in results
    ),
    "before_decision_counts": dict(
        Counter(item["decision"] for item in results)
    ),
    "post_repair_decision_counts": dict(
        Counter(
            item["post_repair_decision"]
            for item in results
        )
    ),
    "repair_action_counts": dict(action_counts),
}

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

print(f"修复处理：{summary['total']} 条")
print(
    f"实际修改：{summary['repaired_document_count']} 条"
)
print(
    f"分数提升：{summary['score_improved_count']} 条"
)
print(
    f"决策变化：{summary['decision_changed_count']} 条"
)
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")
