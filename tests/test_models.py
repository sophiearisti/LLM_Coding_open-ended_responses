from dataclasses import replace
import json

import httpx
from openai import OpenAI
from anthropic import Anthropic
from google.genai import types
import pytest

from model_config import MODELS, GenerationConfig, request_payload, run_fingerprint
import v


def config(model):
    spec = MODELS[model]
    return GenerationConfig(model, spec.default_effort, spec.thinking_modes[0] if spec.thinking_modes else "")


@pytest.mark.parametrize("model", MODELS)
def test_default_request_has_expected_model_and_no_implicit_temperature(model):
    payload = request_payload(config(model), "classify", "actual row")
    assert payload["model"] == model
    assert "temperature" not in payload
    if MODELS[model].provider == "gemini":
        parsed = types.GenerateContentConfig(**payload["config"])
        assert parsed.thinking_config.thinking_level.lower() == config(model).effort
    else:
        assert payload["messages"][-1]["content"] == "actual row"


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol", "gemini-4-argon", "gemini-3.8-flash", "claude-opus-5-5", "claude-sonnet-5-5"])
def test_rejects_temperature_on_reasoning_models(model):
    with pytest.raises(ValueError, match="Temperature"):
        request_payload(config(model), "prompt", "row", 0)


def test_openai_temperature_only_without_reasoning():
    cfg = replace(config("gpt-6-luna"), effort="none")
    assert request_payload(cfg, "p", "r", 0)["temperature"] == 0
    with pytest.raises(ValueError):
        request_payload(replace(cfg, effort="high"), "p", "r", 0)


def test_model_specific_limits():
    for cfg in (replace(config("gpt-6-astra"), effort="none"),
                replace(config("gemini-3.8-flash"), effort="minimal"),
                replace(config("claude-sonnet-5-5"), thinking="between_tools", effort="max"),
                replace(config("claude-opus-5-5"), thinking="disabled")):
        with pytest.raises(ValueError):
            cfg.validate()
    cfg = replace(config("claude-sonnet-5-5"), thinking="between_tools", effort="low")
    assert request_payload(cfg, "p", "r")["thinking"] == {"type": "between_tools"}


def test_legacy_claude_temperature_and_limit():
    cfg = replace(config("claude-sonnet-4-6"), thinking="disabled")
    assert request_payload(cfg, "p", "r", 1)["temperature"] == 1
    with pytest.raises(ValueError):
        cfg.validate(1.2)


def test_run_identity_includes_settings_prompt_and_ordered_data():
    args = [config("gpt-6-luna"), "prompt", '["a","b"]', "text", None, []]
    original = run_fingerprint(*args)
    for index, value in [(0, replace(args[0], effort="high")), (0, config("gpt-6-astra")),
                         (1, "new prompt"), (2, '["b","a"]'), (4, "game"), (5, ["id"])]:
        changed = args.copy()
        changed[index] = value
        assert run_fingerprint(*changed) != original
    assert run_fingerprint(*args) == original
    assert run_fingerprint(*args, run_label="replicate-2") != original


def test_gemini_real_sdk_serialization(monkeypatch):
    from google import genai
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": '{"label":1}'}]}, "finishReason": "STOP"}]})
    monkeypatch.setattr(v, "get_gemini_client", lambda key: genai.Client(api_key=key,
        http_options={"client_args": {"transport": httpx.MockTransport(handle)}}))
    result = v.call_llm_for_message("prompt", "row text", None, "gemini", "fake-key", game=0,
                                   model_override="gemini-4-argon", generation=config("gemini-4-argon"))
    assert json.loads(result) == {"label": 1}
    thinking = captured[0]["generationConfig"]["thinkingConfig"]
    # Protobuf JSON accepts both the proto field name and the lowerCamelCase alias.
    assert thinking.get("thinkingLevel", thinking.get("thinking_level")).lower() == "medium"
    assert "temperature" not in captured[0]["generationConfig"]
    assert "Game 0" in captured[0]["contents"][0]["parts"][0]["text"]


def test_openai_real_sdk_serializes_new_model_and_effort(monkeypatch):
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "gpt-6.1-sol", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"label": 1}'}}]})
    monkeypatch.setattr(v, "get_openai_batch_client", lambda key: OpenAI(api_key=key, http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    result = v.call_llm_for_message("prompt", "TEXT WITHOUT GAME", None, "chatgpt", "fake-key",
                                   model_override="gpt-6.1-sol", generation=config("gpt-6.1-sol"))
    assert json.loads(result) == {"label": 1}
    assert captured[0]["model"] == "gpt-6.1-sol"
    assert captured[0]["reasoning_effort"] == "low"
    assert "TEXT WITHOUT GAME" in captured[0]["messages"][-1]["content"]
    assert "temperature" not in captured[0]


def test_claude_real_sdk_serializes_between_tools():
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
           "content": [{"type": "text", "text": '{"label":1}'}], "stop_reason": "end_turn", "stop_sequence": None,
           "usage": {"input_tokens": 1, "output_tokens": 1}})
    cfg = replace(config("claude-sonnet-5-5"), thinking="between_tools", effort="low")
    with Anthropic(api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as api:
        api.messages.create(**request_payload(cfg, "p", "r"))
    assert captured[0]["thinking"] == {"type": "between_tools"}
    assert captured[0]["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("text, expected", [('{"a": true, "b": null}', {"a": True, "b": None}),
                                          ("```python\n{'a': 1}\n```", {"a": 1})])
def test_parser_handles_json_and_python(text, expected):
    assert v.parse_llm_dict(text) == expected


@pytest.mark.parametrize("text", [None, "", "not a dictionary", "{'a':", "{}"])
def test_invalid_output_is_not_marked_successful(text):
    with pytest.raises(ValueError):
        v.parse_llm_dict(text)
