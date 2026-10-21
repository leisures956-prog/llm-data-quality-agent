import html
import json
from pathlib import Path

from datasets import load_dataset


base_path = Path(
    "data/raw/sample.jsonl"
)

output_path = Path(
    "data/raw/benchmark_v1.jsonl"
)

if output_path.exists():
    raise SystemExit(
        f"{output_path} 已存在，请更换输出文件名"
    )


records = []

for line in base_path.read_text(
    encoding="utf-8"
).splitlines():
    if line.strip():
        item = json.loads(line)

        item.setdefault(
            "source",
            "local_regression",
        )
        item.setdefault(
            "collected_at",
            "2026-10-19",
        )
        item.setdefault(
            "license",
            "internal_demo",
        )

        records.append(item)


targets = [
    ("20231101.zh", "zh", 20),
    ("20231101.en", "en", 20),
]

for config_name, language, target_count in targets:
    dataset = load_dataset(
        "wikimedia/wikipedia",
        config_name,
        split="train",
        streaming=True,
    )

    added_count = 0

    for row in dataset:
        text = (row.get("text") or "").strip()
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()

        if len(text) < 200:
            continue

        paragraphs = [
            part.strip()
            for part in text.splitlines()
            if part.strip()
        ]

        body = "".join(
            f"<p>{html.escape(part)}</p>"
            for part in paragraphs
        )

        records.append(
            {
                "id": (
                    f"wiki_{language}_"
                    f"{added_count + 1:04d}"
                ),
                "url": url,
                "domain": "encyclopedia",
                "language": language,
                "source": (
                    f"wikimedia_wikipedia_"
                    f"{config_name}"
                ),
                "collected_at": "2026-10-19",
                "license": (
                    "CC-BY-SA-3.0-or-GFDL"
                ),
                "title": title,
                "html": (
                    "<html><head>"
                    f"<title>{html.escape(title)}"
                    "</title></head><body>"
                    f"<article>{body}</article>"
                    "</body></html>"
                ),
            }
        )

        added_count += 1

        if added_count >= target_count:
            break

    print(
        f"{config_name} 下载："
        f"{added_count} 条"
    )


output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    "\n".join(
        json.dumps(
            item,
            ensure_ascii=False,
        )
        for item in records
    )
    + "\n",
    encoding="utf-8",
)

print(
    f"最终数据集：{len(records)} 条"
)
print(f"输出文件：{output_path}")