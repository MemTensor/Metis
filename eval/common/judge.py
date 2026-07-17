"""One OpenAI-compatible Judge client shared by all evaluation suites."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_MODEL = "gpt-4.1-mini"


def parse_jsonish(content: str) -> Any:
    content = (content or "").strip()
    if not content:
        raise ValueError("empty_judge_response")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class JudgeClient:
    """Minimal client for APIs implementing OpenAI chat completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: int = 90,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.available = False
        self.status: dict[str, Any] = {
            "requested_model": model,
            "selected_model": model,
            "base_url": self.base_url,
            "available": False,
            "reason": "not_checked",
        }

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("missing_api_key")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def check(self) -> dict[str, Any]:
        if not self.api_key:
            self.status.update({"available": False, "reason": "missing_api_key"})
            return self.status
        try:
            models = self._request("/v1/models")
            raw_items = models.get("data", models if isinstance(models, list) else [])
            ids = []
            for item in raw_items:
                value = item.get("id") or item.get("model") if isinstance(item, dict) else str(item)
                if value:
                    ids.append(value)
            if self.model not in ids:
                self.status.update(
                    {
                        "available": False,
                        "reason": "requested_model_unavailable",
                        "requested_model": self.model,
                        "models_endpoint": "ok",
                        "model_count": len(ids),
                        "available_model_sample": ids[:20],
                    }
                )
                return self.status
            self.status.update(
                {"models_endpoint": "ok", "model_count": len(ids), "selected_model": self.model}
            )
        except Exception as exc:
            self.status.update(
                {
                    "available": False,
                    "reason": "models_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return self.status
        try:
            result = self.chat_json(
                [
                    {"role": "system", "content": "Return compact JSON only."},
                    {"role": "user", "content": 'Return {"ok": true}.'},
                ],
                max_tokens=32,
            )
            if isinstance(result, dict):
                self.available = True
                self.status.update(
                    {
                        "available": True,
                        "reason": "ok",
                        "selected_model": self.model,
                        "judge_temperature": self.temperature,
                    }
                )
        except Exception as exc:
            self.status.update(
                {
                    "available": False,
                    "reason": "chat_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        return self.status

    def chat_json(self, messages: list[dict[str, str]], max_tokens: int = 384) -> Any:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
        }
        try:
            response = self._request("/v1/chat/completions", payload)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            payload.pop("response_format", None)
            response = self._request("/v1/chat/completions", payload)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content and "output_text" in response:
            content = response["output_text"]
        return parse_jsonish(content)


def judge_answer(
    client: JudgeClient,
    *,
    question: str,
    reference: str,
    prediction: str,
) -> dict[str, Any]:
    """Strict semantic equivalence grade; API errors are hard failures."""

    if not client.available:
        raise RuntimeError("judge_required_but_unavailable")
    payload = {
        "question": question,
        "reference_answer": reference,
        "model_answer": prediction,
        "instruction": (
            "Decide whether the model answer is semantically correct. Accept concise "
            "paraphrases, reject contradictions and missing required facts. Return JSON "
            "with keys score (0 or 1), pass (boolean), and rationale (short string)."
        ),
    }
    raw = client.chat_json(
        [
            {"role": "system", "content": "You are a strict answer evaluator. Return JSON only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    )
    score = 1.0 if float(raw.get("score", 1.0 if raw.get("pass") else 0.0)) >= 0.5 else 0.0
    return {
        "score": score,
        "pass": bool(raw.get("pass", score == 1.0)),
        "rationale": str(raw.get("rationale", ""))[:1000],
        "judge_model": client.model,
        "raw_judge": raw,
    }
