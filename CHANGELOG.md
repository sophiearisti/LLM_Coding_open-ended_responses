# Changelog

## 1.0.1 — 2026-10-29

Restores prompt caching, closes an API-key exposure and fixes interruption,
batch-status and cost issues found in review of 1.0.0.

### Security

- Environment API keys are no longer sent to the browser as the key field's value.
  They are used only when the field is empty and `ALLOW_SERVER_KEYS=1` is set
  (see README). Without that variable, enter the key in the interface.

### Performance and cost

- Restore Claude prompt caching, which the shared request builder had dropped:
  the instructions are sent as a cacheable system block (1 h lifetime in batches,
  which can outlive the default 5 minutes and cost more to write). Category creation
  also caches the messages when several temperatures reuse them.
- Send an OpenAI `prompt_cache_key` derived from the prompt. Gemini caches shared
  prefixes implicitly. Prompts shorter than the provider's minimum are not cached.
- Show how many input tokens were read from the cache after each normal run.
- Reuse one API client per key instead of one per row.
- Hash the dataset instead of serializing it on every interface update; rewrite
  run configuration files only when they change.

### Correctness

- A run interrupted by an interface update is saved and resumes without repeating
  finished rows; queued Gemini requests are cancelled instead of discarded.
- Category answers are saved before they are rendered, so an interruption cannot
  discard a paid answer and repeat the request.
- A damaged batch tracking file no longer breaks the whole page, and the batch
  status no longer shows an error after a successful submission.
- Remove a duplicate CSV write in normal mode; stop tracking compiled bytecode.
- Add regression tests for all of the above (64 in total).

## 1.0.0 — 2026-10-29

First numbered release. Updates model configuration and fixes classification
and resume behavior in the previously unversioned Streamlit application.

### Model configuration

- Add GPT-6 Luna/Astra, GPT-6.1 Sol/Terra, Gemini 4 Argon/3.8 Flash/3.5
  Flash-Lite, and Claude Fable 5.1/Opus 5.5/Sonnet 5.5/Haiku 5.5.
- Show model-specific reasoning effort, thinking modes and output token limits.
  Only offer and send temperature when the selected configuration supports it.
- Share request construction between normal and batch processing. Preserve
  selected older text models for comparisons and remove incompatible live models.
- Update Google GenAI to 2.30.1. Use the OpenAI SDK directly, removing unused
  LangChain and tqdm dependencies.

### Correctness and reproducibility

- Always send the input row, including when no game column is selected; preserve
  game value zero. Respect model selection in category creation and assignment.
- Fix the misplaced arguments in normal Claude requests and avoid stale API
  clients after changing credentials.
- Keep the source dataset intact across multiple temperatures.
- Isolate progress by ordered dataset, prompt, model, generation settings and
  column selections. Offer downloadable run configuration alongside results.
- Parse JSON booleans/null and Python dictionaries. Do not count malformed,
  empty or truncated answers as successful classifications.
- Prevent reruns/downloads from automatically repeating failed or completed
  normal/category requests. Explicitly start the same run again to retry failures.
- Prevent duplicate pending batches, isolate batch collection by provider,
  deduplicate collected rows and archive terminal row maps.
- Reset the interface without deleting results or pending batch tracking. Mask
  API key fields and support the environment variable names documented in README.

### Validation

- Automated regression tests and Streamlit interaction tests use fake credentials
  and simulated API responses. No paid live model evaluation is included.
- Existing result files remain on disk. New runs use a new directory layout and
  do not reuse old checkpoints that lack model/configuration identity.
