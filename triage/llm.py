"""A minimal model interface: one call, text in, text out, plus the JSON parsing the agents need.

Providers: Anthropic, AWS Bedrock (Converse API) and Groq. The agents never see a vendor SDK. Choose one with
`--provider` or the LLM_PROVIDER environment variable. `standin` is an offline stand-in, see standin.py.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The provider failed. `rate_limited` is True for quota and throttling errors."""

    def __init__(self, message: str, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 800) -> Reply: ...


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 800) -> Reply:
        if self._client is None and not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("anthropic: ANTHROPIC_API_KEY is not set")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:  # SDK errors differ by version; keep only the class name
            status = getattr(e, "status_code", None)
            where = f" (HTTP {status})" if status else ""
            raise LLMError(f"anthropic: {e.__class__.__name__}{where}", rate_limited=status == 429) from e
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        usage = getattr(response, "usage", None)
        return Reply(
            text,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )


DEFAULT_BEDROCK_MODEL = "us.amazon.nova-lite-v1:0"


class BedrockLLM:
    name = "bedrock"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3  # optional dependency: pip install boto3

            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
            self._client = boto3.client("bedrock-runtime", region_name=region)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 800) -> Reply:
        try:
            response = self.client.converse(
                modelId=self.model,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": max_tokens},
            )
        except Exception as e:
            code = e.response.get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
            throttled = code in ("ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException")
            raise LLMError(f"bedrock: {code or e.__class__.__name__}", rate_limited=throttled) from e
        blocks = response["output"]["message"]["content"]
        text = "".join(b["text"] for b in blocks if "text" in b)
        text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL).strip()
        usage = response.get("usage") or {}
        return Reply(text, int(usage.get("inputTokens", 0) or 0), int(usage.get("outputTokens", 0) or 0))


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    name = "groq"

    def __init__(self, model: str | None = None, client: httpx.Client | None = None):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.client = client or httpx.Client(timeout=60)

    def complete(self, system: str, user: str, max_tokens: int = 800) -> Reply:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("groq: GROQ_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        try:
            response = self.client.post(GROQ_URL, json=payload, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as e:
            raise LLMError(f"groq: {e.__class__.__name__}") from e
        if response.status_code == 429:
            raise LLMError("groq: rate limit reached", rate_limited=True)
        if response.status_code >= 400:
            raise LLMError(f"groq: HTTP {response.status_code}")
        data = response.json()
        usage = data.get("usage") or {}
        text = data["choices"][0]["message"].get("content") or ""
        return Reply(text, int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0))


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    name = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
    if name == "anthropic":
        return AnthropicLLM(model)
    if name == "bedrock":
        return BedrockLLM(model)
    if name == "groq":
        return GroqLLM(model)
    if name == "standin":
        from .standin import ObedientStandIn

        return ObedientStandIn()
    raise ValueError(f"Unknown provider '{name}'. Use anthropic, bedrock, groq or standin.")


def extract_json(text: str) -> dict | None:
    """The first balanced {...} object in the text that parses as JSON. Models wrap JSON in prose or code fences."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


@dataclass
class JsonCall:
    parsed: BaseModel | None
    calls: int
    input_tokens: int
    output_tokens: int


RETRY_NOTE = "\n\nYour previous reply was not a valid JSON object in the required format. Reply with the JSON object only."


def call_json(llm: LLM, system: str, user: str, model_cls: type[T], max_tokens: int = 800, retries: int = 1) -> JsonCall:
    """Ask for JSON and validate it. One retry with a fixed note; the note never repeats the model's or the ticket's text."""
    calls = tin = tout = 0
    prompt = user
    for _ in range(retries + 1):
        reply = llm.complete(system, prompt, max_tokens)  # LLMError propagates to the caller
        calls += 1
        tin += reply.input_tokens
        tout += reply.output_tokens
        data = extract_json(reply.text)
        if data is not None:
            try:
                return JsonCall(model_cls.model_validate(data), calls, tin, tout)
            except ValidationError:
                pass
        prompt = user + RETRY_NOTE
    return JsonCall(None, calls, tin, tout)
