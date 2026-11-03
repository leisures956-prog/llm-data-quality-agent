import json

import pytest

from app.pipeline.llm_judge import (
    ClientResponse,
    JudgeConfig,
    JsonlCache,
    LLMJudgeError,
    MockJudgeClient,
    judge_document,
    select_boundary_documents,
    validate_judge_payload,
)
from tools.evaluate_llm_judge import evaluate_llm_judge


def _document(**overrides):
    document = {
        "id": "doc_001",
        "title": "示例文档",
        "clean_text": "这是一个具有连续正文的示例文档。",
        "clean_text_chars": 1200,
        "paragraph_count": 8,
        "sentence_count": 20,
        "link_ratio": 0.1,
        "production_decision": "review",
        "calibrated_quality_score": 60,
    }
    document.update(overrides)
    return document


def test_validate_judge_payload_accepts_fenced_json():
    result = validate_judge_payload(
        """```json
        {"quality_score": 82, "decision": "keep", "reasons": ["正文完整"],
        "risk_flags": [], "confidence": 0.9}
        ```""",
        expected_document_id="doc_001",
    )

    assert result["document_id"] == "doc_001"
    assert result["quality_score"] == 82
    assert result["decision"] == "keep"
    assert result["confidence"] == 0.9


def test_validate_judge_payload_rejects_invalid_decision():
    with pytest.raises(LLMJudgeError, match="decision 无效"):
        validate_judge_payload(
            {
                "quality_score": 82,
                "decision": "maybe",
                "reasons": [],
                "risk_flags": [],
                "confidence": 0.9,
            },
            expected_document_id="doc_001",
        )


def test_mock_judge_returns_contract_result():
    result = judge_document(
        _document(),
        JudgeConfig(max_retries=0),
        MockJudgeClient(),
    )

    assert result["status"] == "ok"
    assert result["decision"] in {"keep", "review", "reject"}
    assert 0 <= result["quality_score"] <= 100
    assert result["cached"] is False
    assert "prompt_version" in result


def test_judge_uses_cache_on_second_call(tmp_path):
    class CountingClient:
        def __init__(self):
            self.calls = 0

        def complete(self, document, messages, config):
            del document, messages, config
            self.calls += 1
            return ClientResponse(
                content={
                    "quality_score": 70,
                    "decision": "keep",
                    "reasons": ["test"],
                    "risk_flags": [],
                    "confidence": 0.8,
                },
                usage={"total_tokens": 3},
            )

    client = CountingClient()
    cache = JsonlCache(tmp_path / "cache.jsonl")
    config = JudgeConfig(max_retries=0)

    first = judge_document(_document(), config, client, cache=cache)
    second = judge_document(_document(), config, client, cache=cache)

    assert client.calls == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert json.loads((tmp_path / "cache.jsonl").read_text())[
        "result"
    ]["decision"] == "keep"


def test_judge_retries_invalid_response():
    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def complete(self, document, messages, config):
            del document, messages, config
            self.calls += 1
            if self.calls == 1:
                return ClientResponse(
                    content={"decision": "invalid"},
                    usage={},
                )
            return ClientResponse(
                content={
                    "quality_score": 50,
                    "decision": "review",
                    "reasons": ["边界样本"],
                    "risk_flags": [],
                    "confidence": 0.6,
                },
                usage={},
            )

    client = FlakyClient()
    result = judge_document(
        _document(),
        JudgeConfig(max_retries=1, retry_backoff_seconds=0),
        client,
    )

    assert client.calls == 2
    assert result["status"] == "ok"
    assert result["retry_count"] == 1


def test_select_boundary_documents_uses_review_and_score_range():
    documents = [
        _document(id="keep_doc", production_decision="keep", calibrated_quality_score=85),
        _document(id="review_doc", production_decision="review", calibrated_quality_score=75),
        _document(id="border_doc", production_decision="keep", calibrated_quality_score=55),
        _document(id="reject_doc", production_decision="reject", calibrated_quality_score=10),
    ]
    selected = select_boundary_documents(
        documents,
        JudgeConfig(
            selection={
                "include_decisions": ["review"],
                "score_min": 40,
                "score_max": 70,
            }
        ),
    )

    assert [item["id"] for item in selected] == [
        "review_doc",
        "border_doc",
    ]


def test_evaluate_llm_judge_preserves_review_as_separate_decision():
    results = [
        {
            "id": "doc_001",
            "title": "保留文档",
            "llm_judge": {
                "status": "ok",
                "decision": "keep",
                "quality_score": 80,
                "confidence": 0.9,
                "reasons": [],
                "risk_flags": [],
            },
        },
        {
            "id": "doc_002",
            "title": "待复核文档",
            "llm_judge": {
                "status": "ok",
                "decision": "review",
                "quality_score": 55,
                "confidence": 0.6,
                "reasons": [],
                "risk_flags": [],
            },
        },
    ]
    labels = [
        {
            "id": "doc_001",
            "split": "dev",
            "reviewed": True,
            "human_decision": "keep",
        },
        {
            "id": "doc_002",
            "split": "dev",
            "reviewed": True,
            "human_decision": "reject",
        },
    ]

    report = evaluate_llm_judge(results, labels, split="dev")

    assert report["evaluated_count"] == 2
    assert report["coverage"] == 1.0
    assert report["machine_decision_counts"] == {
        "keep": 1,
        "review": 1,
    }
    assert report["review_count"] == 1
    assert report["automatic_decision_count"] == 1
