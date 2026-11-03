"""Optional LLM-as-a-Judge support for boundary documents.

The module is deliberately independent from the production scoring rules.  It
provides a stable JSON contract, a deterministic mock client for tests, an
OpenAI-compatible HTTP client, and a small JSONL cache so an experiment can be
re-run without sending the same document twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx


VALID_DECISIONS = {"keep", "review", "reject"}
DEFAULT_JUDGE_VERSION = "llm_judge_v1"
DEFAULT_PROMPT_VERSION = "quality_judge_prompt_v1"


class LLMJudgeError(RuntimeError):
    """Raised when a judge response cannot be obtained or validated."""


@dataclass(frozen=True)
class JudgeConfig:
    judge_version: str = DEFAULT_JUDGE_VERSION
    prompt_version: str = DEFAULT_PROMPT_VERSION
    model: str = "llm-judge-model"
    endpoint_env: str = "LLM_JUDGE_ENDPOINT"
    api_key_env: str = "LLM_JUDGE_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.2
    max_chars: int = 12000
    selection: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JudgeConfig":
        retry = payload.get("retry", {})
        selection = payload.get("selection", {})
        return cls(
            judge_version=str(
                payload.get("judge_version", DEFAULT_JUDGE_VERSION)
            ),
            prompt_version=str(
                payload.get("prompt_version", DEFAULT_PROMPT_VERSION)
            ),
            model=str(payload.get("model", "llm-judge-model")),
            endpoint_env=str(
                payload.get("endpoint_env", "LLM_JUDGE_ENDPOINT")
            ),
            api_key_env=str(
                payload.get("api_key_env", "LLM_JUDGE_API_KEY")
            ),
            timeout_seconds=float(
                payload.get("timeout_seconds", 30.0)
            ),
            max_retries=int(retry.get("max_retries", 2)),
            retry_backoff_seconds=float(
                retry.get("backoff_seconds", 0.2)
            ),
            max_chars=int(payload.get("max_chars", 12000)),
            selection=dict(selection),
        )


@dataclass(frozen=True)
class ClientResponse:
    content: Any
    usage: dict[str, int]


class LLMJudgeClient(Protocol):
    def complete(
        self,
        document: dict[str, Any],
        messages: list[dict[str, str]],
        config: JudgeConfig,
    ) -> ClientResponse:
        """Return the model content and optional token usage."""


def load_judge_config(path: Path) -> JudgeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"LLM Judge 配置必须是 JSON 对象：{path}")
    return JudgeConfig.from_dict(payload)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _document_id(document: dict[str, Any]) -> str:
    value = document.get("id", document.get("document_id", ""))
    return str(value).strip()


def _machine_decision(document: dict[str, Any]) -> str:
    for key in (
        "production_decision",
        "calibrated_decision",
        "decision",
        "content_decision",
    ):
        value = document.get(key)
        if value in VALID_DECISIONS:
            return str(value)
    return "unknown"


def _machine_score(document: dict[str, Any]) -> float | None:
    for key in (
        "calibrated_quality_score",
        "quality_score",
        "post_repair_quality_score",
    ):
        value = document.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _trim_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[文本已截断，仅用于评估提示词]"


def build_messages(
    document: dict[str, Any],
    config: JudgeConfig,
) -> list[dict[str, str]]:
    document_payload = {
        "document_id": _document_id(document),
        "title": str(document.get("title", "")),
        "url": str(document.get("url", "")),
        "language": str(document.get("language", "")),
        "domain": str(document.get("domain", "")),
        "clean_text": _trim_text(
            document.get("clean_text", ""),
            config.max_chars,
        ),
        "machine_decision": _machine_decision(document),
        "machine_score": _machine_score(document),
        "machine_reasons": document.get(
            "calibrated_reasons",
            document.get("reasons", []),
        ),
    }
    system = (
        "你是网页数据质量评审员。只根据提供的页面信息判断数据是否适合 "
        "进入高质量语言模型数据集。必须只返回一个 JSON 对象，不要输出 Markdown。"
        "JSON 字段必须是 quality_score、decision、reasons、risk_flags、confidence。"
        "decision 只能是 keep、review、reject；quality_score 为 0 到 100 的整数；"
        "confidence 为 0 到 1 的数字。注意区分知识正文、导航页、聚合页和模板内容。"
    )
    user = (
        f"prompt_version={config.prompt_version}\n"
        "请评估下面的文档，并返回固定 JSON：\n"
        + json.dumps(document_payload, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def make_cache_key(
    document: dict[str, Any],
    config: JudgeConfig,
) -> str:
    payload = {
        "judge_version": config.judge_version,
        "prompt_version": config.prompt_version,
        "model": config.model,
        "max_chars": config.max_chars,
        "document_id": _document_id(document),
        "title": document.get("title", ""),
        "clean_text": document.get("clean_text", ""),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        payload = content
    elif isinstance(content, str):
        text = content.strip()
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            text = fenced.group(1).strip()
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMJudgeError("LLM 返回内容不是合法 JSON") from exc
    else:
        raise LLMJudgeError("LLM 返回内容类型无效")

    if not isinstance(payload, dict):
        raise LLMJudgeError("LLM 返回的 JSON 必须是对象")
    return payload


def validate_judge_payload(
    content: Any,
    expected_document_id: str,
) -> dict[str, Any]:
    payload = _parse_json_content(content)
    decision = payload.get("decision")
    if decision not in VALID_DECISIONS:
        raise LLMJudgeError(
            f"LLM decision 无效：{decision!r}"
        )

    score = payload.get("quality_score")
    if isinstance(score, bool):
        raise LLMJudgeError("quality_score 不能是布尔值")
    try:
        score_value = float(score)
    except (TypeError, ValueError) as exc:
        raise LLMJudgeError("quality_score 必须是数字") from exc
    if not 0 <= score_value <= 100:
        raise LLMJudgeError("quality_score 必须位于 0 到 100")

    reasons = payload.get("reasons")
    risk_flags = payload.get("risk_flags")
    if not isinstance(reasons, list) or not all(
        isinstance(item, str) for item in reasons
    ):
        raise LLMJudgeError("reasons 必须是字符串数组")
    if not isinstance(risk_flags, list) or not all(
        isinstance(item, str) for item in risk_flags
    ):
        raise LLMJudgeError("risk_flags 必须是字符串数组")

    confidence = payload.get("confidence")
    if isinstance(confidence, bool):
        raise LLMJudgeError("confidence 不能是布尔值")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise LLMJudgeError("confidence 必须是数字") from exc
    if not 0 <= confidence_value <= 1:
        raise LLMJudgeError("confidence 必须位于 0 到 1")

    response_document_id = payload.get("document_id")
    if response_document_id not in (None, "", expected_document_id):
        raise LLMJudgeError("LLM 返回的 document_id 与输入不一致")

    return {
        "document_id": expected_document_id,
        "quality_score": int(round(score_value)),
        "decision": decision,
        "reasons": reasons,
        "risk_flags": risk_flags,
        "confidence": round(confidence_value, 4),
    }


class JsonlCache:
    """Small append-friendly cache keyed by the deterministic cache key."""

    def __init__(self, path: Path):
        self.path = path
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM cache JSONL 解析失败：{self.path}:{line_number}"
                ) from exc
            cache_key = row.get("cache_key")
            result = row.get("result")
            if cache_key and isinstance(result, dict):
                self._records[str(cache_key)] = result

    def get(self, cache_key: str) -> dict[str, Any] | None:
        result = self._records.get(cache_key)
        return dict(result) if result is not None else None

    def put(self, cache_key: str, result: dict[str, Any]) -> None:
        self._records[cache_key] = dict(result)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {"cache_key": key, "result": value},
                ensure_ascii=False,
            )
            for key, value in sorted(self._records.items())
        ]
        self.path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )


class MockJudgeClient:
    """Deterministic client used by tests and the offline pilot."""

    def complete(
        self,
        document: dict[str, Any],
        messages: list[dict[str, str]],
        config: JudgeConfig,
    ) -> ClientResponse:
        del messages, config
        decision = _machine_decision(document)
        score = _machine_score(document)
        if decision == "unknown":
            text_chars = _safe_int(document.get("clean_text_chars"))
            if text_chars < 100:
                decision = "reject"
                score = 20
            elif text_chars >= 3000:
                decision = "keep"
                score = 80
            else:
                decision = "review"
                score = 60

        if score is None:
            score = {"keep": 80, "review": 60, "reject": 20}[decision]

        risk_flags: list[str] = []
        if document.get("navigation_like"):
            risk_flags.append("navigation_like")
        if document.get("aggregation_like"):
            risk_flags.append("aggregation_like")

        return ClientResponse(
            content={
                "document_id": _document_id(document),
                "quality_score": int(round(score)),
                "decision": decision,
                "reasons": ["mock judge result for contract testing"],
                "risk_flags": risk_flags,
                "confidence": 0.5,
            },
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )


class HttpLLMJudgeClient:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key

    def complete(
        self,
        document: dict[str, Any],
        messages: list[dict[str, str]],
        config: JudgeConfig,
    ) -> ClientResponse:
        del document
        endpoint = self.endpoint or os.getenv(config.endpoint_env, "")
        if not endpoint:
            raise LLMJudgeError(
                f"未设置 LLM Judge endpoint，请配置环境变量 {config.endpoint_env}"
            )
        api_key = self.api_key or os.getenv(config.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": config.model,
            "temperature": 0,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        try:
            response = httpx.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMJudgeError(f"LLM 请求失败：{exc}") from exc

        try:
            message_content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMJudgeError("LLM 响应缺少 choices.message.content") from exc

        usage_payload = body.get("usage", {})
        usage = {
            "prompt_tokens": _safe_int(
                usage_payload.get("prompt_tokens")
            ),
            "completion_tokens": _safe_int(
                usage_payload.get("completion_tokens")
            ),
            "total_tokens": _safe_int(
                usage_payload.get("total_tokens")
            ),
        }
        return ClientResponse(content=message_content, usage=usage)


def judge_document(
    document: dict[str, Any],
    config: JudgeConfig,
    client: LLMJudgeClient,
    cache: JsonlCache | None = None,
) -> dict[str, Any]:
    document_id = _document_id(document)
    if not document_id:
        raise LLMJudgeError("文档缺少 id")

    cache_key = make_cache_key(document, config)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            cached["cached"] = True
            cached["latency_ms"] = 0
            return cached

    messages = build_messages(document, config)
    started = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            response = client.complete(document, messages, config)
            normalized = validate_judge_payload(
                response.content,
                expected_document_id=document_id,
            )
            result = {
                **normalized,
                "status": "ok",
                "judge_version": config.judge_version,
                "prompt_version": config.prompt_version,
                "model": config.model,
                "cache_key": cache_key,
                "cached": False,
                "retry_count": attempt,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "prompt_tokens": response.usage.get(
                    "prompt_tokens",
                    0,
                ),
                "completion_tokens": response.usage.get(
                    "completion_tokens",
                    0,
                ),
                "total_tokens": response.usage.get(
                    "total_tokens",
                    0,
                ),
            }
            if cache is not None:
                cache.put(cache_key, result)
                cache.save()
            return result
        except Exception as exc:  # retry both transport and schema failures
            last_error = exc
            if attempt >= config.max_retries:
                break
            if config.retry_backoff_seconds > 0:
                time.sleep(config.retry_backoff_seconds * (attempt + 1))

    raise LLMJudgeError(
        f"文档 {document_id} 的 LLM Judge 失败：{last_error}"
    ) from last_error


def select_boundary_documents(
    documents: list[dict[str, Any]],
    config: JudgeConfig,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    if include_all:
        return list(documents)

    selection = config.selection or {}
    include_decisions = set(
        selection.get(
            "include_decisions",
            ["review"],
        )
    )
    score_min = float(selection.get("score_min", 40))
    score_max = float(selection.get("score_max", 70))
    include_near_duplicate = bool(
        selection.get("include_near_duplicate", True)
    )
    selected = []
    for document in documents:
        decision = _machine_decision(document)
        score = _machine_score(document)
        score_boundary = (
            score is not None
            and score_min <= score <= score_max
        )
        near_duplicate = bool(
            document.get("is_near_duplicate")
            or document.get("duplicate_status") == "near_duplicate"
        )
        decision_disagreement = (
            document.get("production_decision")
            and document.get("calibrated_decision")
            and document.get("production_decision")
            != document.get("calibrated_decision")
        )
        if (
            decision in include_decisions
            or score_boundary
            or (include_near_duplicate and near_duplicate)
            or decision_disagreement
        ):
            selected.append(document)
    return selected


def summarize_results(
    results: list[dict[str, Any]],
    *,
    input_path: str = "",
    mode: str = "mock",
    config: JudgeConfig | None = None,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    cache_hit_count = 0
    total_tokens = 0
    total_latency_ms = 0.0
    for result in results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if result.get("decision") in VALID_DECISIONS:
            decision = str(result["decision"])
            decision_counts[decision] = (
                decision_counts.get(decision, 0) + 1
            )
        cache_hit_count += bool(result.get("cached"))
        total_tokens += _safe_int(result.get("total_tokens"))
        total_latency_ms += _safe_float(result.get("latency_ms"))

    summary: dict[str, Any] = {
        "summary_version": "llm_judge_summary_v1",
        "input_path": input_path,
        "mode": mode,
        "total": len(results),
        "status_counts": status_counts,
        "decision_counts": decision_counts,
        "cache_hit_count": cache_hit_count,
        "error_count": status_counts.get("error", 0),
        "total_tokens": total_tokens,
        "total_latency_ms": round(total_latency_ms, 2),
        "average_latency_ms": round(
            total_latency_ms / max(len(results), 1),
            2,
        ),
    }
    if config is not None:
        summary.update(
            {
                "judge_version": config.judge_version,
                "prompt_version": config.prompt_version,
                "model": config.model,
            }
        )
    return summary
