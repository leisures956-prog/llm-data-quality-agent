import json

from tools.compare_benchmark_runs import compare_runs


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            for row in rows
        ),
        encoding="utf-8",
    )


def test_compare_detects_decision_and_score_change(tmp_path):
    old_file = tmp_path / "old.jsonl"
    new_file = tmp_path / "new.jsonl"

    write_jsonl(
        old_file,
        [
            {
                "id": "doc_001",
                "title": "MDN Blog",
                "production_decision": "keep",
                "calibrated_quality_score": 90,
                "calibrated_reasons": [
                    "正文内容密度较高"
                ],
                "quality_rule_version": "v2",
            }
        ],
    )

    write_jsonl(
        new_file,
        [
            {
                "id": "doc_001",
                "title": "MDN Blog",
                "production_decision": "reject",
                "calibrated_quality_score": 65,
                "calibrated_reasons": [
                    "正文内容密度较高",
                    "页面疑似博客或文章聚合页",
                ],
                "quality_rule_version": "v3",
            }
        ],
    )

    report = compare_runs(old_file, new_file)

    assert report["decision_delta"] == {
        "keep": -1,
        "reject": 1,
    }

    assert len(report["decision_changes"]) == 1
    assert report["decision_changes"][0]["id"] == "doc_001"

    assert report["score_changes"][0]["score_delta"] == -25

    assert report["reason_changes"][0]["added_reasons"] == [
        "页面疑似博客或文章聚合页"
    ]


def test_compare_same_file_has_no_changes(tmp_path):
    file_path = tmp_path / "same.jsonl"

    write_jsonl(
        file_path,
        [
            {
                "id": "doc_001",
                "production_decision": "keep",
                "calibrated_quality_score": 75,
            }
        ],
    )

    report = compare_runs(file_path, file_path)

    assert report["decision_changes"] == []
    assert report["score_changes"] == []
    assert report["reason_changes"] == []