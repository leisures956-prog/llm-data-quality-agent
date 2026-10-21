import json
from pathlib import Path

from app.pipeline.sampling import (
    assign_sampling_weights,
    summarize_sampling,
)


input_path = Path(
    "data/processed/"
    "classified_benchmark_v1.jsonl"
)

output_path = Path(
    "data/processed/"
    "sampling_plan_benchmark_v1.jsonl"
)

summary_path = Path(
    "reports/"
    "sampling_benchmark_summary.json"
)


documents = [
    json.loads(line)
    for line in input_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

results = assign_sampling_weights(
    documents
)

summary = summarize_sampling(results)

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

print(f"采样计划生成：{len(results)} 条")
print(
    f"可采样："
    f"{summary['eligible_count']} 条"
)
print(
    f"排除："
    f"{summary['excluded_count']} 条"
)
print(
    f"总采样权重："
    f"{summary['total_sampling_weight']}"
)
print(f"结果文件：{output_path}")
print(f"统计文件：{summary_path}")