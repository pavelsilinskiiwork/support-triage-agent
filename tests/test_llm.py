import json
from types import SimpleNamespace

import pytest

from triage.llm import AnthropicLLM, BedrockLLM, GroqLLM, LLMError, call_json, extract_json, get_llm
from triage.models import Classification

from .helpers import ScriptedLLM, classify


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": {"b": "}"}}\n```\nDone') == {"a": {"b": "}"}}
    assert extract_json("no json here") is None
    assert extract_json('{"broken": ') is None
    assert extract_json('{bad} {"ok": true}') == {"ok": True}


def test_call_json_retries_once_with_a_fixed_note():
    llm = ScriptedLLM("not json", classify())
    call = call_json(llm, "sys", "user text", Classification)
    assert call.parsed.category == "billing" and call.calls == 2
    assert llm.calls[1][1].startswith("user text") and "valid JSON" in llm.calls[1][1]
    assert call.input_tokens == 200


def test_call_json_gives_up_after_the_retry():
    call = call_json(ScriptedLLM("nope", "still nope"), "s", "u", Classification)
    assert call.parsed is None and call.calls == 2


def test_call_json_rejects_wrong_types():
    bad = json.dumps({"category": "billing", "confidence": 7})
    assert call_json(ScriptedLLM(bad, bad), "s", "u", Classification).parsed is None


def test_unknown_category_becomes_other():
    assert Classification.model_validate({"category": "weather", "confidence": 0.5}).category == "other"


def test_anthropic_adapter_with_a_fake_client():
    seen = {}

    class Messages:
        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")],
                                   usage=SimpleNamespace(input_tokens=11, output_tokens=3))

    llm = AnthropicLLM("m", client=SimpleNamespace(messages=Messages()))
    r = llm.complete("sys", "usr", 50)
    assert (r.text, r.input_tokens, r.output_tokens) == ("hello", 11, 3)
    assert seen["system"] == "sys" and seen["messages"] == [{"role": "user", "content": "usr"}]


def test_anthropic_without_key_is_a_clean_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM().complete("s", "u")


def test_anthropic_rate_limit_is_flagged():
    class E(Exception):
        status_code = 429

    class Messages:
        def create(self, **kw):
            raise E()

    with pytest.raises(LLMError) as err:
        AnthropicLLM("m", client=SimpleNamespace(messages=Messages())).complete("s", "u")
    assert err.value.rate_limited


def test_bedrock_adapter_strips_thinking():
    class Client:
        def converse(self, **kw):
            return {"output": {"message": {"content": [{"text": "<thinking>hm</thinking>answer"}]}},
                    "usage": {"inputTokens": 5, "outputTokens": 2}}

    r = BedrockLLM("m", client=Client()).complete("s", "u")
    assert (r.text, r.input_tokens) == ("answer", 5)


def test_groq_adapter(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")

    class Http:
        def post(self, url, json, headers):
            return SimpleNamespace(status_code=200, json=lambda: {
                "choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 4, "completion_tokens": 1}})

    assert GroqLLM("m", client=Http()).complete("s", "u").text == "hi"


def test_get_llm():
    assert get_llm("standin").name == "standin"
    assert get_llm("anthropic", "x").model == "x"
    with pytest.raises(ValueError):
        get_llm("nope")
