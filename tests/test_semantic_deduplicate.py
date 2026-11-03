import numpy as np
import pytest

from app.pipeline.semantic_deduplicate import (
    annotate_semantic_duplicates,
)


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, **kwargs):
        return np.asarray([self.vectors[text] for text in texts])


def test_semantic_duplicate_detection_with_injected_encoder():
    documents = [
        {"id": "a", "clean_text": "改写后的数据质量文章"},
        {"id": "b", "clean_text": "另一种表达的数据质量文章"},
        {"id": "c", "clean_text": "完全不同的天气内容"},
    ]
    encoder = FakeEncoder(
        {
            "改写后的数据质量文章": [1.0, 0.0],
            "另一种表达的数据质量文章": [0.99, 0.1],
            "完全不同的天气内容": [0.0, 1.0],
        }
    )

    results, pairs = annotate_semantic_duplicates(
        documents,
        threshold=0.9,
        encoder=encoder,
    )

    assert len(pairs) == 1
    assert pairs[0]["id_a"] == "a"
    assert pairs[0]["id_b"] == "b"
    assert results[0]["is_semantic_duplicate"] is True
    assert results[1]["semantic_duplicate_count"] == 2
    assert results[2]["is_semantic_duplicate"] is False


def test_semantic_threshold_is_validated():
    with pytest.raises(ValueError, match="threshold"):
        annotate_semantic_duplicates([], threshold=1.1)


def test_empty_or_single_document_has_no_pairs():
    results, pairs = annotate_semantic_duplicates(
        [{"id": "a", "clean_text": "内容"}],
        threshold=0.8,
    )

    assert pairs == []
    assert results[0]["is_semantic_duplicate"] is False
