import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import v
from model_config import GenerationConfig


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.fixture
def ui(monkeypatch, tmp_path):
    fake = MagicMock()
    fake.session_state = State(stop_requested=False, batch_status=False)
    fake.columns.return_value = [MagicMock(), MagicMock()]
    fake.button.return_value = False
    monkeypatch.setattr(v, "st", fake)
    monkeypatch.setattr(v, "RESULTS_PATH", str(tmp_path))
    return fake


def info(model="gpt-6-luna", effort="none", thinking=""):
    cfg = GenerationConfig(model, effort, thinking)
    return {"proveedor": cfg.spec.provider, "modelo": model, "generation": cfg,
            "run_id": "test", "run_folder": "test", "prompt": "prompt"}


def test_multiple_temperatures_keep_input_and_resume(ui, monkeypatch, tmp_path):
    call = MagicMock(return_value='{"label":1}')
    monkeypatch.setattr(v, "call_llm_for_message", call)
    df = pd.DataFrame({"text": ["first", "second"]})
    v.ejecutar_procesamiento_st(df, "p", info(), [0, 1], "text", "test", "fake")
    assert [c.args[1] for c in call.call_args_list] == ["first", "second", "first", "second"]
    assert all(c.args[-1].model == "gpt-6-luna" for c in call.call_args_list)
    v.ejecutar_procesamiento_st(df, "p", info(), [0, 1], "text", "test", "fake")
    assert call.call_count == 4
    assert len(list(tmp_path.rglob("*.metadata.json"))) == 2


def test_claude_normal_path_passes_key_and_selected_model(ui, monkeypatch):
    call = MagicMock(return_value='{"label":1}')
    monkeypatch.setattr(v, "call_llm_for_message", call)
    cfg = info("claude-opus-5-5", "medium", "adaptive")
    v.ejecutar_procesamiento_st(pd.DataFrame({"text": ["hi"]}), "p", cfg, [None], "text", "test", "correct-key")
    assert call.call_args.args[4] == "correct-key"
    assert call.call_args.args[6] == "claude-opus-5-5"


def test_failed_row_retried_only_on_explicit_run(ui, monkeypatch, tmp_path):
    call = MagicMock(side_effect=['{"label":1}', 'invalid'])
    monkeypatch.setattr(v, "call_llm_for_message", call)
    df = pd.DataFrame({"text": ["first", "second"]})
    v.ejecutar_procesamiento_st(df, "p", info(), [0], "text", "test", "fake")
    v.ejecutar_procesamiento_st(df, "p", info(), [0], "text", "test", "fake")
    assert call.call_count == 2
    ui.session_state["attempted_outputs"] = set()
    call.side_effect = None
    call.return_value = '{"label":0}'
    v.ejecutar_procesamiento_st(df, "p", info(), [0], "text", "test", "fake")
    assert call.call_count == 3
    assert call.call_args.args[1] == "second"
    assert len(pd.read_csv(next(tmp_path.rglob("*.csv")))) == 2


def test_categories_respect_selected_model_and_download_does_not_repeat(ui, monkeypatch):
    call = MagicMock(return_value='{"A":"definition"}')
    monkeypatch.setattr(v, "call_llm_for_message", call)
    df = pd.DataFrame({"text": ["hello", "world"]})
    cfg = info("gpt-6-astra", "high")
    for _ in range(2):
        v.ejecutar_procesamiento_crear_st(df, "p", cfg, [None], "text", "fake")
    assert call.call_count == 1
    assert call.call_args.kwargs["model_override"] == "gpt-6-astra"
    assert "hello" in call.call_args.args[1]


@pytest.mark.parametrize("provider, model, effort, thinking", [("chatgpt", "gpt-6-astra", "high", ""), ("claude", "claude-sonnet-5-5", "low", "between_tools")])
def test_batch_uses_shared_parameters_and_deduplicates_pending(ui, monkeypatch, tmp_path, provider, model, effort, thinking):
    api = MagicMock()
    api.messages.batches.create.return_value.id = "batch-test"
    api.batches.create.return_value.id = "batch-test"
    api.files.create.return_value.id = "file-test"
    monkeypatch.setattr(v, "get_openai_batch_client", lambda key: api)
    monkeypatch.setattr(v, "get_claude_client", lambda key: api)
    cfg = info(model, effort, thinking)
    df = pd.DataFrame({"text": ["hello"]})
    v.ejecutar_batch_st(df, "p", cfg, [None], "text", "test", "fake")
    if provider == "chatgpt":
        body = json.loads(next(tmp_path.rglob("*.jsonl")).read_text())["body"]
        assert body["reasoning_effort"] == effort
    else:
        body = api.messages.batches.create.call_args.kwargs["requests"][0]["params"]
        assert body["thinking"] == {"type": thinking}
        assert body["output_config"]["effort"] == effort
    assert body["model"] == model and "temperature" not in body
    ui.session_state.batch_status = False
    v.ejecutar_batch_st(df, "p", cfg, [None], "text", "test", "fake")
    assert (api.batches.create if provider == "chatgpt" else api.messages.batches.create).call_count == 1


def test_batch_collection_deduplicates_and_skips_invalid(ui, monkeypatch, tmp_path):
    provider = "chatgpt"
    output = tmp_path / "results.csv"
    meta = {"provider": provider, "output_path": str(output), "temperature": None, "total_rows": 2}
    mapping = {"a": {"idx": 0, "message": "one"}, "b": {"idx": 1, "message": "two"}}
    Path(v._st_batch_status_path(provider)).write_text(json.dumps({"batch-test": meta}))
    Path(v._rowmap_path(provider, "batch-test")).write_text(json.dumps(mapping))
    def row(id, text):
        return json.dumps({"custom_id": id, "response": {"status_code": 200, "body": {
            "choices": [{"finish_reason": "stop", "message": {"content": text}}]}}})
    api = MagicMock()
    api.batches.retrieve.return_value.status = "completed"
    api.batches.retrieve.return_value.output_file_id = "out"
    api.files.content.return_value.text = row("a", '{"label":1}') + "\n" + row("b", "bad")
    monkeypatch.setattr(v, "get_openai_batch_client", lambda key: api)
    v._collect_batch_results_st("batch-test", meta, "fake")
    assert pd.read_csv(output)["row_id"].tolist() == [0]
    assert not v.load_st_batch_status(provider)
    assert (tmp_path / provider / "completed_batch-test.json").exists()
    v.write_rows_to_csv(str(output), [{"row_id": 0, "label": 2}])
    assert len(pd.read_csv(output)) == 1


def test_model_controls_follow_capabilities():
    script = '''
import streamlit as st
from v import seleccionar_llm_st, configuracion_temperaturas
info, key = seleccionar_llm_st()
configuracion_temperaturas(False, info["generation"])
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert not app.multiselect
    app.selectbox(key="effort_gpt-6-luna_").select("none").run()
    assert len(app.multiselect) == 1
    app.selectbox[0].select("Gemini").run()
    assert not app.multiselect
    assert app.selectbox(key="effort_gemini-3.8-flash_").options == ["low", "medium", "high"]
    app.selectbox[0].select("Claude").run()
    app.selectbox(key="thinking_claude-sonnet-5-5").select("between_tools").run()
    assert app.selectbox(key="effort_claude-sonnet-5-5_between_tools").options == ["low", "medium", "high"]
    assert not app.exception


def test_main_smoke_and_reset_keeps_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sentinel = tmp_path / "Results" / "saved.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("keep")
    app = AppTest.from_file(str(Path(v.__file__))).run()
    assert not app.exception
    app.button[0].click().run()
    assert sentinel.read_text() == "keep"


def test_full_assignment_flow_does_not_call_api_again_on_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "RESULTS_PATH", str(tmp_path))
    call = MagicMock(return_value='{"category":1}')
    monkeypatch.setattr(v, "call_llm_for_message", call)
    script = '''
import streamlit as st
import pandas as pd
from v import menu_st
for k in ["proceso_finalizado", "stop_requested", "batch_status"]:
    st.session_state.setdefault(k, False)
menu_st(pd.DataFrame({"text":["sample one", "sample two"]}), 2)
'''
    app = AppTest.from_string(script).run()
    app.text_input(key="api_key_chatgpt").set_value("fake-key").run()
    app.text_area(key="txt_Classification").set_value("Return a dictionary with category 0 or 1.").run()
    next(b for b in app.button if b.label == "Generate Prompt and Run").click().run(timeout=10)
    assert not app.exception
    assert call.call_count == 2
    app.run()
    assert not app.exception
    assert call.call_count == 2
