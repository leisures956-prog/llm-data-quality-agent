from app.pipeline.near_deduplicate import (
    annotate_near_duplicates,
)


def test_near_duplicate_detection():
    documents = [
        {
            "id": "doc_a",
            "clean_text": (
                "高质量网页数据需要经过正文抽取、"
                "重复内容清理和质量评分。"
                "这些步骤可以帮助构建大语言模型预训练数据集。"
                "数据清洗能够降低噪声并提高训练数据质量。"
            ),
        },
        {
            "id": "doc_b",
            "clean_text": (
                "高质量网页数据需要经过正文抽取、"
                "重复内容清理与质量评分。"
                "这些步骤可以帮助构建大语言模型预训练数据集。"
                "数据清洗能够降低噪声并提升训练数据质量。"
            ),
        },
        {
            "id": "doc_c",
            "clean_text": "今天的天气非常晴朗。",
        },
    ]

    results, pairs = annotate_near_duplicates(
        documents,
        threshold=0.75,
    )

    assert len(pairs) >= 1
    assert results[0]["is_near_duplicate"] is True
    assert results[1]["is_near_duplicate"] is True
    assert results[2]["is_near_duplicate"] is False