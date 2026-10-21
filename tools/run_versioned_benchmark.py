import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_NAME = os.environ.get(
    "BENCHMARK_RUN_NAME",
    "benchmark_v1_run_v2",
)

DATASET_PATH = Path(
    os.environ.get(
        "BENCHMARK_DATASET_PATH",
        "data/raw/benchmark_v1.jsonl",
    )
)

LABELS_PATH_VALUE = os.environ.get(
    "BENCHMARK_LABELS_PATH",
    "data/labels/benchmark_audit_v1.jsonl",
).strip()

LABELS_PATH = (
    Path(LABELS_PATH_VALUE)
    if LABELS_PATH_VALUE
    else None
)

AUDIT_MODE = os.environ.get(
    "BENCHMARK_ENABLE_AUDIT",
    "auto",
).strip().lower()

if AUDIT_MODE in {
    "0",
    "false",
    "no",
    "off",
}:
    INCLUDE_AUDIT = False
elif AUDIT_MODE in {
    "1",
    "true",
    "yes",
    "on",
}:
    INCLUDE_AUDIT = True
else:
    INCLUDE_AUDIT = LABELS_PATH is not None

PROCESSED_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / RUN_NAME
)

REPORT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / RUN_NAME
)

CORE_MODULES = [
    "app.pipeline.run_dataset_check",
    "app.pipeline.run_features",
    "app.pipeline.run_deduplicate",
    "app.pipeline.run_near_deduplicate",
    "app.pipeline.run_scoring",
]

def build_modules() -> list[str]:
    modules = list(CORE_MODULES)

    if INCLUDE_AUDIT:
        modules.append(
            "tools.evaluate_benchmark_audit"
        )

    modules.append(
        "tools.run_calibrated_benchmark"
    )

    if INCLUDE_AUDIT:
        modules.append(
            "tools.evaluate_calibrated_benchmark_audit"
        )

    modules.extend(
        [
            "tools.run_finalize_benchmark",
            "tools.run_distribution_benchmark",
            "tools.run_sampling_benchmark",
        ]
    )

    if INCLUDE_AUDIT:
        modules.extend(
            [
                "tools.run_ablation_benchmark_audit",
                "tools.build_benchmark_report",
            ]
        )

    return modules


def copy_tree_files(
    source_dir: Path,
    target_dir: Path,
) -> None:
    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source_path in source_dir.rglob("*"):
        relative_path = source_path.relative_to(
            source_dir
        )
        target_path = target_dir / relative_path

        if source_path.is_dir():
            target_path.mkdir(
                parents=True,
                exist_ok=True,
            )
        else:
            target_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                source_path,
                target_path,
            )

def rewrite_path_value(value):
    if isinstance(value, dict):
        return {
            key: rewrite_path_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            rewrite_path_value(item)
            for item in value
        ]

    if not isinstance(value, str):
        return value

    normalized = value.replace(
        "/",
        "\\",
    )

    if normalized.startswith(
        "reports\\"
    ):
        suffix = normalized[
            len("reports\\") :
        ]
        return str(
            Path("reports")
            / RUN_NAME
            / Path(suffix)
        )

    if normalized.startswith(
        "data\\processed\\"
    ):
        suffix = normalized[
            len("data\\processed\\") :
        ]
        return str(
            Path("data")
            / "processed"
            / RUN_NAME
            / Path(suffix)
        )

    return value


def rewrite_report_file(
    report_path: Path,
) -> None:
    if report_path.suffix == ".json":
        payload = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        report_path.write_text(
            json.dumps(
                rewrite_path_value(payload),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    elif report_path.suffix == ".jsonl":
        output_lines = []

        for line in report_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue

            payload = json.loads(line)

            output_lines.append(
                json.dumps(
                    rewrite_path_value(payload),
                    ensure_ascii=False,
                )
            )

        report_path.write_text(
            "\n".join(output_lines)
            + "\n",
            encoding="utf-8",
        )


def rewrite_report_paths(
    report_dir: Path,
) -> None:
    for report_path in report_dir.rglob("*"):
        if report_path.suffix in {
            ".json",
            ".jsonl",
        }:
            rewrite_report_file(
                report_path
            )


def build_lightweight_report(
    report_dir: Path,
) -> None:
    report_names = [
        "dataset_manifest.json",
        "near_duplicate_pairs.jsonl",
        "quality_summary.json",
        "calibrated_benchmark_summary.json",
        "decision_source_benchmark_summary.json",
        "distribution_benchmark_summary.json",
        "sampling_benchmark_summary.json",
    ]

    bundle = {
        "benchmark_version": RUN_NAME,
        "dataset_path": str(DATASET_PATH),
        "audit_enabled": False,
        "reports": {},
        "missing_reports": [],
    }

    for report_name in report_names:
        report_path = report_dir / report_name

        if not report_path.exists():
            bundle["missing_reports"].append(
                str(report_path)
            )
            continue

        relative_report_path = (
            Path("reports")
            / RUN_NAME
            / report_name
        )

        if report_name.endswith(".jsonl"):
            records = [
                json.loads(line)
                for line in report_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            bundle["reports"][report_name] = {
                "path": str(relative_report_path),
                "record_count": len(records),
                "records": records,
            }
        else:
            bundle["reports"][report_name] = {
                "path": str(relative_report_path),
                "content": json.loads(
                    report_path.read_text(
                        encoding="utf-8"
                    )
                ),
            }

    manifest_report = bundle[
        "reports"
    ].get(
        "dataset_manifest.json",
        {},
    )

    manifest_content = manifest_report.get(
        "content",
        {},
    )

    bundle["provenance"] = manifest_content.get(
        "provenance",
        {},
    )

    bundle["dataset_metadata"] = (
        manifest_content.get(
            "summary",
            {},
        ).get(
            "metadata",
            {},
        )
    )

    bundle["source_registry"] = manifest_content.get(
        "summary",
        {},
    ).get(
        "source_registry",
        {},
    )

    output_path = (
        report_dir
        / f"{RUN_NAME}_run_report.json"
    )

    output_path.write_text(
        json.dumps(
            bundle,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )



def main() -> None:
    if INCLUDE_AUDIT and LABELS_PATH is None:
        raise ValueError(
            "已启用人工审核，但未提供 BENCHMARK_LABELS_PATH"
        )

    required_files = [
        PROJECT_ROOT / DATASET_PATH,
    ]

    if INCLUDE_AUDIT and LABELS_PATH is not None:
        required_files.append(
            PROJECT_ROOT / LABELS_PATH
        )

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到必需文件：{path}"
            )

    if PROCESSED_OUTPUT.exists():
        raise FileExistsError(
            "处理结果目录已存在，为避免覆盖，程序停止："
            f"{PROCESSED_OUTPUT}"
        )

    if REPORT_OUTPUT.exists():
        raise FileExistsError(
            "报告目录已存在，为避免覆盖，程序停止："
            f"{REPORT_OUTPUT}"
        )

    with tempfile.TemporaryDirectory(
        prefix="llmqa_benchmark_"
    ) as temporary_directory:
        temporary_root = Path(
            temporary_directory
        )

        for folder_name in [
            "app",
            "tools",
            "config",
        ]:
            shutil.copytree(
                PROJECT_ROOT / folder_name,
                temporary_root / folder_name,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    ".pytest_cache",
                ),
            )

        input_paths = [DATASET_PATH]

        if INCLUDE_AUDIT and LABELS_PATH is not None:
            input_paths.append(LABELS_PATH)

        for relative_path in input_paths:
            target_path = (
                temporary_root / relative_path
            )
            target_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                PROJECT_ROOT / relative_path,
                target_path,
            )


        for output_dir in [
            temporary_root
            / "data"
            / "processed",
            temporary_root
            / "reports",
        ]:
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        environment = os.environ.copy()
        environment["DATASET_PATH"] = (
            DATASET_PATH.as_posix()
        )
        environment["DATASET_VERSION"] = RUN_NAME
        environment["PIPELINE_VERSION"] = (
            "quality_pipeline_v1"
        )
        environment["RUN_ID"] = RUN_NAME
        environment.pop(
            "DATASET_REPORT_PATH",
            None,
        )

        for module_name in build_modules():
            print(f"\n===== {module_name} =====")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    module_name,
                ],
                cwd=temporary_root,
                env=environment,
                check=True,
            )

        temporary_processed = (
            temporary_root
            / "data"
            / "processed"
        )

        temporary_reports = (
            temporary_root
            / "reports"
        )

        copy_tree_files(
            temporary_processed,
            PROCESSED_OUTPUT,
        )

        copy_tree_files(
            temporary_reports,
            REPORT_OUTPUT,
        )

        rewrite_report_paths(
            REPORT_OUTPUT
        )

        if not INCLUDE_AUDIT:
            build_lightweight_report(
                REPORT_OUTPUT
            )

    print("\n版本化 Benchmark 运行完成")
    print(
        f"处理结果：{PROCESSED_OUTPUT}"
    )
    print(
        f"报告结果：{REPORT_OUTPUT}"
    )
    print(
        "原有 data/processed 和 reports 下的通用文件未被覆盖"
    )


if __name__ == "__main__":
    main()
