from app.pipeline.repair import repair_document, repair_text


def test_repeated_lines_are_removed():
    text = "第一段内容。\n第一段内容。\n第二段内容。"

    repaired, actions = repair_text(text)

    assert repaired == "第一段内容。\n第二段内容。"
    assert "remove_repeated_line" in actions


def test_boilerplate_line_is_removed():
    text = "广告 广告 广告\n这是正文内容。"

    repaired, actions = repair_text(text)

    assert repaired == "这是正文内容。"
    assert "remove_boilerplate_line" in actions


def test_meaningful_line_is_preserved():
    document = {
        "id": "doc",
        "clean_text": "数据质量评估可以降低训练数据噪声。",
    }

    result = repair_document(document)

    assert result["clean_text"] == document["clean_text"]
    assert result["repair_applied"] is False
    assert result["repair_actions"] == []
