## Model-aware configuration and reliable classification

This first numbered release updates the text-model catalog for OpenAI, Gemini
and Claude and applies each model's supported reasoning and sampling parameters
to both normal and batch requests.

- **OpenAI:** GPT-6 Luna/Astra and GPT-6.1 Sol/Terra, plus retained GPT-5 models.
- **Gemini:** Gemini 4 Argon, 3.8 Flash and 3.5 Flash-Lite, plus selected previews.
- **Claude:** Fable 5.1 and Opus/Sonnet/Haiku 5.5, plus retained 4.x models.
- Model-specific reasoning/thinking controls, output token limits and conditional
  temperature selection. Incompatible parameter combinations are blocked.
- Fixes for missing input text, ignored model selections, Claude request
  arguments and multi-temperature processing.
- Configuration-aware checkpoints, downloadable run metadata, independent run
  labels, safer retries and batch collection. Reset preserves saved work.

### Install or upgrade

Extract the attached application ZIP into a new directory and create a Python
3.12+ virtual environment. From the extracted directory:

```bash
python -m pip install -r tool/requirements.txt
python -m streamlit run tool/v.py
```

Google GenAI is upgraded to 2.30.1. LangChain is no longer required.
The application ZIP excludes the repository's historical virtual environment.

### Validation and migration

49 automated tests passed, including Streamlit interaction tests and actual SDK
serialization against simulated HTTP responses. No paid live model evaluation
was performed; model access depends on your provider account. Validate coding
quality against human labels before changing a research baseline.

Existing results are preserved. New checkpoints use dataset, prompt, model and
configuration identity, so old checkpoints are not automatically resumed.
See `CHANGELOG.md` and `docs/model-support.md` for details and official sources.
