"""Explicit text-model capabilities. Sources and review date: docs/model-support.md."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

APP_VERSION = "1.0.0"
CATALOG_DATE = "2026-10-29"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    efforts: tuple[str, ...]
    default_effort: str
    temperature: str = "never"  # never, always, or none (reasoning disabled)
    thinking_modes: tuple[str, ...] = ()
    max_tokens: int = 32768  # conservative app cap, not the provider's full limit


MODELS = {
    "gpt-6-luna": ModelSpec("chatgpt", ("none",) + EFFORTS, "low", "none"),
    "gpt-6.1-sol": ModelSpec("chatgpt", EFFORTS, "low"),
    "gpt-6.1-terra": ModelSpec("chatgpt", ("none",) + EFFORTS, "low", "none"),
    "gpt-6-astra": ModelSpec("chatgpt", EFFORTS, "low"),
    "gpt-5.4-mini": ModelSpec("chatgpt", ("none", "low", "medium", "high", "xhigh"), "none", "none"),
    "gpt-5.4": ModelSpec("chatgpt", ("none", "low", "medium", "high", "xhigh"), "none", "none"),
    "gpt-5.2": ModelSpec("chatgpt", ("none", "low", "medium", "high", "xhigh"), "none", "none"),
    "gemini-3.8-flash": ModelSpec("gemini", ("low", "medium", "high"), "medium"),
    "gemini-4-argon": ModelSpec("gemini", ("low", "medium", "high"), "medium"),
    "gemini-3.5-flash-lite": ModelSpec("gemini", ("minimal", "low", "medium", "high"), "minimal"),
    "gemini-3.1-pro-preview": ModelSpec("gemini", ("low", "medium", "high"), "high"),
    "gemini-3-flash-preview": ModelSpec("gemini", ("minimal", "low", "medium", "high"), "high"),
    "claude-sonnet-5-5": ModelSpec("claude", EFFORTS, "high", thinking_modes=("adaptive", "between_tools")),
    "claude-haiku-5-5": ModelSpec("claude", EFFORTS, "medium", thinking_modes=("adaptive",)),
    "claude-opus-5-5": ModelSpec("claude", EFFORTS, "medium", thinking_modes=("adaptive",)),
    "claude-fable-5-1": ModelSpec("claude", EFFORTS, "high", thinking_modes=("adaptive",)),
    "claude-sonnet-4-6": ModelSpec("claude", ("low", "medium", "high", "max"), "high", "disabled", ("adaptive", "disabled")),
    "claude-opus-4-6": ModelSpec("claude", ("low", "medium", "high", "max"), "high", "disabled", ("adaptive", "disabled")),
    "claude-haiku-4-5-20251001": ModelSpec("claude", (), "", "always", ("disabled",)),
}


def models_for(provider):
    return [name for name, spec in MODELS.items() if spec.provider == provider]


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    effort: str
    thinking: str = ""
    max_tokens: int = 8192

    @property
    def spec(self):
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        return MODELS[self.model]

    @property
    def allows_temperature(self):
        rule = self.spec.temperature
        return rule == "always" or (rule == "none" and self.effort == "none") or (rule == "disabled" and self.thinking == "disabled")

    def validate(self, temperature=None):
        spec = self.spec
        if self.effort not in spec.efforts and (spec.efforts or self.effort):
            raise ValueError(f"Unsupported reasoning effort for {self.model}: {self.effort}")
        if self.thinking not in spec.thinking_modes and (spec.thinking_modes or self.thinking):
            raise ValueError(f"Unsupported thinking mode for {self.model}: {self.thinking}")
        if self.thinking == "between_tools" and self.effort not in ("low", "medium", "high"):
            raise ValueError("between_tools requires low, medium, or high effort.")
        if type(self.max_tokens) is not int or not 256 <= self.max_tokens <= spec.max_tokens:
            raise ValueError(f"Output token limit must be 256–{spec.max_tokens}.")
        if temperature is not None:
            if not self.allows_temperature:
                raise ValueError("Temperature is not available for this model/configuration.")
            upper = 1 if spec.provider == "claude" else 2
            if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= upper:
                raise ValueError(f"Temperature must be between 0 and {upper}.")

    def metadata(self, temperature=None):
        self.validate(temperature)
        return {**asdict(self), "provider": self.spec.provider, "temperature": temperature,
                "app_version": APP_VERSION, "catalog_date": CATALOG_DATE}


def request_payload(config, prompt, user_message, temperature=None):
    """The same payload builder is used for normal requests and batch requests."""
    config.validate(temperature)
    provider = config.spec.provider
    params = {"model": config.model}
    if temperature is not None:
        params["temperature"] = temperature
    if provider == "chatgpt":
        params.update(max_completion_tokens=config.max_tokens, reasoning_effort=config.effort,
                      messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_message}])
    elif provider == "claude":
        params.update(max_tokens=config.max_tokens, system=prompt,
                      messages=[{"role": "user", "content": user_message}], thinking={"type": config.thinking})
        if config.effort:
            params["output_config"] = {"effort": config.effort}
    else:
        params.update(contents=user_message, config={"system_instruction": prompt,
                      "max_output_tokens": config.max_tokens,
                      "thinking_config": {"thinking_level": config.effort}})
    return params


def run_fingerprint(config, prompt, dataset_json, message_col, game_col, keep_cols, run_label="run-1"):
    """Include ordered data and prompt so row IDs cannot resume a different dataset."""
    content = {"config": config.metadata(), "prompt": prompt, "data": dataset_json,
               "message_col": message_col, "game_col": game_col, "keep_cols": keep_cols,
               "run_label": run_label}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
