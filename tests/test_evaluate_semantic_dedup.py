import numpy as np
import pytest

from tools.evaluate_semantic_dedup import evaluate_pair_methods


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, **kwargs):
        return np.asarray([self.vectors[text] for text in texts])


def pair(pair_id, text_a, text_b, gold_label, pair_type):
    return {
        "pair_id": pair_id,
        "text_a": text_a,
        "text_b": text_b,
        "gold_label": gold_label,
        "pair_type": pair_type,
    }


def test_pair_evaluation_reports_method_metrics_and_hybrid():
    rows = [
        pair("p1", "same a", "same b", "duplicate", "paraphrase"),
        pair("p2", "different a", "different b", "non_duplicate", "negative"),
    ]
    encoder = FakeEncoder(
        {
            "same a": [1.0, 0.0],
            "same b": [0.99, 0.1],
            "different a": [1.0, 0.0],
            "different b": [0.0, 1.0],
        }
    )

    report = evaluate_pair_methods(
        rows,
        char_threshold=0.9,
        embedding_threshold=0.9,
        encoder=encoder,
    )

    assert report["pair_count"] == 2
    assert report["metrics"]["embedding"]["true_positive"] == 1
    assert report["metrics"]["embedding"]["false_positive"] == 0
    assert report["metrics"]["embedding"]["f1"] == 1.0
    assert "hybrid_intersection" in report["metrics"]
    assert len(report["details"]) == 2


def test_pair_evaluation_rejects_invalid_gold_label():
    rows = [
        pair("p1", "a", "b", "unknown", "invalid"),
    ]
    encoder = FakeEncoder({"a": [1.0, 0.0], "b": [1.0, 0.0]})

    with pytest.raises(ValueError, match="gold_label"):
        evaluate_pair_methods(rows, encoder=encoder)


def test_pair_evaluation_rejects_duplicate_pair_ids():
    rows = [
        pair("p1", "a", "b", "duplicate", "exact"),
        pair("p1", "c", "d", "duplicate", "exact"),
    ]
    encoder = FakeEncoder(
        {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [1.0, 0.0], "d": [1.0, 0.0]}
    )

    with pytest.raises(ValueError, match="重复 pair_id"):
        evaluate_pair_methods(rows, encoder=encoder)
