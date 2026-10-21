import json
from pathlib import Path


report_dir = Path("reports")
output_path = report_dir / "benchmark_v1_run_report.json"

report_files = {
    "dataset_manifest": "dataset_manifest.json",
    "near_duplicate_pairs": "near_duplicate_pairs.jsonl",
    "quality_summary": "quality_summary.json",
    "audit_raw": "benchmark_audit_summary.json",
    "audit_calibrated": (
        "benchmark_audit_calibrated_summary.json"
    ),
    "calibrated_summary": (
        "calibrated_benchmark_summary.json"
    ),
    "decision_summary": (
        "decision_source_benchmark_summary.json"
    ),
    "distribution_summary": (
        "distribution_benchmark_summary.json"
    ),
    "sampling_summary": (
        "sampling_benchmark_summary.json"
    ),
}


bundle = {
    "benchmark_version": "benchmark_v1",
    "reports": {},
    "missing_reports": [],
}


for name, filename in report_files.items():
    path = report_dir / filename

    if not path.exists():
        bundle["missing_reports"].append(
            str(path)
        )
        continue

    if filename.endswith(".jsonl"):
        records = [
            json.loads(line)
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        bundle["reports"][name] = {
            "path": str(path),
            "record_count": len(records),
            "records": records,
        }
    else:
        bundle["reports"][name] = {
            "path": str(path),
            "content": json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            ),
        }


output_path.write_text(
    json.dumps(
        bundle,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print("benchmark_v1 总报告生成完成")
print(f"包含报告：{len(bundle['reports'])}")
print(f"缺失报告：{len(bundle['missing_reports'])}")
print(f"输出文件：{output_path}")