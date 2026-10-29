# Model configuration in 1.0.0

Catalog checked against official documentation on 2026-10-29. Account access,
quotas and provider lifecycle changes can still restrict a listed model.
`tool/model_config.py` is the single source of capability rules used by the UI,
normal requests and batch requests. Unknown models are rejected rather than
assigned guessed capabilities. This is a text classification catalog; live,
image, audio and restricted-access models are intentionally excluded.

| Provider / model | Reasoning control | Temperature |
| --- | --- | --- |
| OpenAI GPT-6 Astra, GPT-6.1 Sol | low, medium, high, xhigh, max | Omitted |
| OpenAI GPT-6 Luna, GPT-6.1 Terra | none, low, medium, high, xhigh, max | Only with none |
| OpenAI GPT-5.4, GPT-5.4 mini, GPT-5.2 | none, low, medium, high, xhigh | Only with none |
| Gemini 4 Argon, 3.8 Flash | low, medium, high | Omitted |
| Gemini 3.5 Flash-Lite, 3 Flash preview | minimal, low, medium, high | Provider default, omitted |
| Gemini 3.1 Pro preview | low, medium, high | Provider default, omitted |
| Claude Fable 5.1, Opus 5.5, Haiku 5.5 | Adaptive thinking; low, medium, high, xhigh, max effort | Omitted |
| Claude Sonnet 5.5 | Adaptive thinking; low, medium, high, xhigh, max effort. Also between_tools at low–high | Omitted |
| Claude Sonnet 4.6, Opus 4.6 | Adaptive or disabled; low, medium, high, max effort | Only when thinking is disabled, 0–1 |
| Claude Haiku 4.5 | Thinking disabled in this app | 0–1 |

The selected model is used for both category creation and assignment. Defaults
are GPT-6 Luna at low effort, Gemini 3.8 Flash at medium, and Claude Sonnet 5.5
at high. These are starting settings, not evidence of equivalent reasoning or
classification quality across providers. Compare each configuration against
your human-coded validation sample before replacing an experimental baseline.
Legacy entries remain available for comparisons; old result files are preserved
but are not automatically resumed by the new configuration-aware run layout.

Output budgets include reasoning tokens. The application defaults to 8,192 and
allows 256–32,768, a deliberately conservative application cap. Truncated or
empty answers are rejected and the row remains retryable. Lower effort or raise
the token limit when reasoning exhausts the budget before a final answer.

## API routing

- OpenAI: Chat Completions with `reasoning_effort` and
  `max_completion_tokens`; batch uses the same body and endpoint. This app does
  not use tools, so migrating tool calls to Responses is not required.
- Gemini: `generate_content` with `thinking_config.thinking_level`,
  `system_instruction` and `max_output_tokens`. Google still documents this
  endpoint as legacy. The SDK is updated to 2.30.1 for Argon support.
- Claude: Messages with `thinking`, `output_config.effort` and `max_tokens`.
  Normal requests stream internally and extract only final text blocks;
  batch requests use the same generation parameters.

The app exposes a curated set of controls. It does not expose manual legacy
thinking budgets, tool settings, top-p/top-k, service tiers or provider pricing.
Gemini batch submission remains outside this release.

## Official references

- [OpenAI model catalog](https://developers.openai.com/api/docs/models)
- [OpenAI GPT-6 migration and compatibility](https://developers.openai.com/api/docs/guides/latest-model)
- [OpenAI GPT-5.4 guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4)
- [OpenAI GPT-5.2 guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2)
- [Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini thinking levels](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
- [Gemini Argon migration and SDK requirements](https://ai.google.dev/gemini-api/docs/generate-content/latest-model)
- [Gemini sampling parameter restrictions](https://ai.google.dev/gemini-api/docs/troubleshooting)
- [Claude models](https://platform.claude.com/docs/en/models/overview)
- [Claude effort levels](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Claude Sonnet 5.5 compatibility](https://platform.claude.com/docs/en/models/sonnet-5-5/overview)
- [Claude Opus 5.5 migration](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)
- [Claude Haiku 5.5 migration](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide)

## Validation and limits

Automated tests cover capability validation, actual SDK request serialization
with mocked HTTP, Streamlit controls, both processing paths, multi-temperature
runs, category creation, invalid responses, retry/resume and batch collection.
Tests do not send research data to providers or validate account-specific model
access. A live, paid comparison against human labels has not been run.
