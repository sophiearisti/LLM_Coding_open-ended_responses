import streamlit as st
import pandas as pd
import ast
import re
import json
import hashlib
import threading
from functools import lru_cache
from model_config import APP_VERSION, MODELS, GenerationConfig, models_for, request_payload, run_fingerprint
from pathlib import Path
from google import genai
from utils import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
from openai import OpenAI


def seleccionar_llm_st():
    st.markdown("### Model Configuration")
    locked = st.session_state.get("proceso_finalizado", False)
    provider_label = st.selectbox("Provider", ["ChatGPT", "Gemini", "Claude"], disabled=locked)
    provider = provider_label.lower()
    model = st.selectbox("Model", models_for(provider), key=f"model_{provider}", disabled=locked)
    spec = MODELS[model]
    names = {"chatgpt": ("OPENAI_API_KEY", "OAI_2"), "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI"),
             "claude": ("ANTHROPIC_API_KEY", "CLAUDE")}
    api_key = st.text_input("API key", type="password", key=f"api_key_{provider}", disabled=locked)
    # Server-side keys never reach the browser; they are used only if the operator opts in.
    if not api_key and os.getenv("ALLOW_SERVER_KEYS") == "1":
        api_key = next((os.getenv(name) for name in names[provider] if os.getenv(name)), "")
        if api_key:
            st.caption("Using the server's API key.")
    thinking = ""
    if spec.thinking_modes:
        thinking = st.selectbox("Thinking mode", spec.thinking_modes, key=f"thinking_{model}", disabled=locked)
    efforts = spec.efforts
    if thinking == "between_tools":
        efforts = tuple(e for e in efforts if e in ("low", "medium", "high"))
    effort = ""
    if efforts:
        effort = st.selectbox("Thinking level" if provider == "gemini" else "Reasoning effort", efforts,
                              index=efforts.index(spec.default_effort), key=f"effort_{model}_{thinking}", disabled=locked,
                              help="Higher effort can increase latency and token usage. Compare quality on your own coded examples.")
    max_tokens = st.number_input("Maximum output tokens (including reasoning)", min_value=256,
                                max_value=spec.max_tokens, value=8192, step=256, key=f"tokens_{model}", disabled=locked)
    config = GenerationConfig(model, effort, thinking, int(max_tokens))
    st.caption("Model availability depends on your API account. Model catalog verified on 2026-10-29.")
    return {"proveedor": provider, "modelo": model, "generation": config}, api_key


# 2. Definimos la función que pone la variable en False
def desactivar_algo():
    st.session_state.proceso_finalizado = False
    
def input_prompt_component(label, help_text, example_text, esta_bloqueado):
    # We create a visual container for each section
    with st.container(border=True):
        col_tit, col_help = st.columns([0.8, 0.2])
        
        with col_tit:
            st.markdown(f"### {label}")
            
        with col_help:
            # We replace the button/info with a Popover (floating window)
            with st.popover("💡 HELP"):
                st.markdown(f"**Example of {label}:**")
                st.caption(example_text) # More legible text for examples

        # 3. Pasamos la función al radio button
        method = st.radio(
            f"How would you like to enter the {label}?",
            ["Write text", "Upload .txt file"],
            key=f"radio_{label}",
            horizontal=True, 
            disabled=esta_bloqueado,
            on_change=desactivar_algo 
        )
        if method == "Write text":
            return st.text_area(
                f"Enter the {label}:", 
                placeholder=help_text, 
                key=f"txt_{label}",
                height=150, # Fixed height to prevent it from looking too small,
                disabled = esta_bloqueado,
                on_change=desactivar_algo 
            )
        else:
            file = st.file_uploader(
                f"Upload the file for {label}", 
                type="txt", 
                key=f"file_{label}",
                disabled = esta_bloqueado,
                on_change=desactivar_algo 
            )
            if file:
                return file.getvalue().decode("utf-8")
    return ""

def input_prompt_format_component(label, help_text, example_text, esta_bloqueado, type):
    # We create a visual container for each section
    with st.container(border=True):
        col_tit, col_help, col_warning = st.columns([0.5, 0.2, 0.3])
        
        with col_tit:
            st.markdown(f"### {label}")
            
        with col_help:
            # We replace the button/info with a Popover (floating window)
            with st.popover("💡 HELP"):
                st.markdown(f"**Example of {label}:**")
                st.caption(example_text) # More legible text for examples
                
        with col_warning:
            
            with st.popover("⚠️ WARNING"):
                st.markdown(f"**Warning for {label}:**")
                st.caption("Make sure the structure is in a JSON format, otherwise it will cause errors.")


        # 3. Pasamos la función al radio button
        method = st.radio(
            f"How would you like to enter the {label}?",
            ["Write text", "Upload .txt file", "Pre-structured JSON"],
            key=f"radio_{label}",
            horizontal=True, 
            disabled=esta_bloqueado,
            on_change=desactivar_algo 
        )
        
        if method == "Write text":
            return st.text_area(
                f"Enter the {label}:", 
                placeholder=help_text, 
                key=f"txt_{label}",
                height=150, # Fixed height to prevent it from looking too small,
                disabled = esta_bloqueado,
                on_change=desactivar_algo 
            )
        elif method == "Upload .txt file":
            file = st.file_uploader(
                f"Upload the file for {label}", 
                type="txt", 
                key=f"file_{label}",
                disabled = esta_bloqueado,
                on_change=desactivar_algo 
            )
            if file:
                return file.getvalue().decode("utf-8")
        elif method == "Pre-structured JSON":
            # 1. Primer bloque de texto (Prefijo)
            prefix = st.text_area("Initial Context (Text):", key=f"prefix_{label}", height=100, disabled = esta_bloqueado)

            st.markdown("---")
            st.write("🔧 **Data Structure**")

            # 2. Inicialización con las 3 filas por defecto (A, B, C)
            key_rows = f"rows_{label}"
            
            if key_rows not in st.session_state and type == "create":
                st.session_state[key_rows] = [
                    {"key": "A", "value": "the category is A: <definition of the category>"},
                    {"key": "B", "value": "the category is B: <definition of the category>"},
                    {"key": "C", "value": "the category is C: <definition of the category>"}
                ]
            elif key_rows not in st.session_state:
                st.session_state[key_rows] = [
                    {"key": "A", "value": "0"},
                    {"key": "B", "value": "1"},
                    {"key": "C", "value": "1"}
                ]
            
            # Contenedor para las categorías
            for i, row in enumerate(st.session_state[key_rows]):
                col1, col_eq, col2, col_btn = st.columns([0.2, 0.05, 0.65, 0.1]) # Ajusté los anchos para que la definición tenga más espacio
                
                with col1:
                    st.session_state[key_rows][i]["key"] = st.text_input(
                        f"Clave {i}", value=row["key"], label_visibility="collapsed", key=f"k_{label}_{i}", disabled = esta_bloqueado
                    )
                with col_eq:
                    st.markdown("<h4 style='text-align: center; margin-top: 0;'> = </h4>", unsafe_allow_html=True)
                with col2:
                    st.session_state[key_rows][i]["value"] = st.text_input(
                        f"Valor {i}", value=row["value"], label_visibility="collapsed", key=f"v_{label}_{i}", disabled = esta_bloqueado
                    )
                with col_btn:
                    # El botón de eliminar funciona perfectamente si deciden quitar A, B o C
                    if st.button("🗑️", key=f"del_{label}_{i}", disabled = esta_bloqueado):
                        st.session_state[key_rows].pop(i)
                        st.rerun()

            
            # Botón para agregar una nueva categoría (D, E, etc.)
            if st.button("➕ Add Category", key=f"add_{label}", disabled = esta_bloqueado):
                st.session_state[key_rows].append({"key": "", "value": ""})
                st.rerun()

            st.markdown("---")

            # 3. Segundo bloque de texto (Sufijo)
            suffix = st.text_area("Final Instructions (Text):", key=f"suffix_{label}", height=100, disabled = esta_bloqueado)

            # Construcción interna del formato estructurado
            json_part = {r["key"]: r["value"] for r in st.session_state[key_rows] if r["key"]}
            full_content = f"{prefix}\n{json.dumps(json_part, ensure_ascii=False)}\n{suffix}"
            return full_content
                           
    return ""

def menu_st(df, menu_type):
    
    estrategia = None
    
    if menu_type == 2:
        st.markdown("---")
        st.subheader("Assignment Strategy")

        # Selección de estrategia
        estrategia = st.radio(
            "Select the prompting strategy::",
            ["Zero-Shot", "Few-Shot", "Zero-Shot CoT", "Few-Shot CoT"],
            horizontal=True, disabled=st.session_state.get("proceso_finalizado", False)
        )
        
    elif menu_type == 1:
        st.markdown("---")
        st.subheader("Categories Creation Strategy")

    info_llm, API_KEY = seleccionar_llm_st()

    if info_llm and API_KEY:
        # Always show pending batches, regardless of whether a file is uploaded
        crear_prompt_obtener_resultados(info_llm, df, API_KEY, estrategia, menu_type)
        
        mostrar_estado_batches_st(API_KEY, info_llm["proveedor"])
             
def crear_prompt_obtener_resultados(info_llm, df, API_KEY, estrategia, menu_type):
        
    # Creamos una variable corta para no escribir tanto
    esta_bloqueado = st.session_state.proceso_finalizado # Si el proceso se ha marcado como finalizado, bloqueamos los inputs
    
    st.write(f" **Configured:** {info_llm['proveedor']} ({info_llm['modelo']})")
    st.header("Prompt Configuration")

    # --- BLOQUES BÁSICOS ---
    rol = input_prompt_component("Role", "e.g., You are an economics expert...", HELP_ROLE, esta_bloqueado)
    contexto = input_prompt_component("Context", "e.g., This data comes from...", HELP_CONTEXTO, esta_bloqueado)
    
    if menu_type == 2:
        clasificacion = input_prompt_component("Classification", "e.g., Classify into A, B, or C...", HELP_CLASIFICACION, esta_bloqueado)
        formato = input_prompt_format_component("Format", "e.g., Return a JSON...",  HELP_FORMAT, esta_bloqueado, type="classify")
        constraints = input_prompt_component("Constraints", "e.g., Do not use adjectives...", HELP_CONSTRAINTS, esta_bloqueado)
    else:
        clasificacion = input_prompt_component("Create categories task", "e.g., Classify into A, B, or C...", HELP_CLASIFICACION_CAT, esta_bloqueado)
        formato = input_prompt_format_component("Format", "e.g., Return a JSON...",  HELP_FORMAT_CAT, esta_bloqueado, type="create")
        constraints = input_prompt_component("Constraints", "e.g., Do not use adjectives...", HELP_CONSTRAINTS, esta_bloqueado)

    extra_content = ""

    # --- BLOQUES DINÁMICOS (Corregidos) ---
    # Usamos 'in' correctamente para detectar la estrategia
    # si estrategia no es None
    if estrategia:
        
        if "Few-Shot" in estrategia:
            help_text = HELP_FS_COT if "CoT" in estrategia else HELP_FS
            extra_content += "\n" + input_prompt_component("Examples", "Add examples of input/output...", help_text, esta_bloqueado)
        
        # Si la estrategia es "Zero-Shot CoT" o "Few-Shot CoT"
        if "CoT" in estrategia:
            extra_content += "\n" + input_prompt_component("Chain of Thought (CoT)", "Reasoning instructions...", HELP_COT, esta_bloqueado)    
        
    # --- CONFIGURACIÓN EXTRA ---
    st.info("Columns Configuration")
    
    message_col = st.selectbox(
        "Select the column containing the messages/texts:",
        options=df.columns,
        help="This is the column the LLM will read for classification.",
        disabled=esta_bloqueado
    )
    
    game_col, keep_cols = None, []
    if estrategia:
        game_col_options = ["(none)"] + list(df.columns)
        game_col_sel = st.selectbox(
            "Select the game column (optional):",
            options=game_col_options,
            help="If your data has a game column, the LLM will receive that context.",
            disabled=esta_bloqueado
        )
        
        game_col = None if game_col_sel == "(none)" else game_col_sel

        keep_cols = st.multiselect(
            "Columns to keep in output (for merging):",
            options=[c for c in df.columns if c != message_col],
            help="These columns will be copied from the original dataset into each result row so you can merge later.",
            disabled=esta_bloqueado
        )

    st.info("Generation Configuration")
    temps = configuracion_temperaturas(esta_bloqueado, info_llm["generation"])
    
    # --- PROCESSING MODE (only for Claude / GPT) ---
    proveedor = info_llm['proveedor']
    
    # Parameters are validated by the shared model capability registry.
    
    if proveedor in ("claude", "chatgpt") and estrategia:
        st.info("Processing Mode")
        processing_mode = st.radio(
            "How should the LLM process the rows?",
            ["Normal (line by line)", "Batch API (async, ~50% cheaper, up to 24 h)"],
            key="processing_mode_radio",
            horizontal=True,
            disabled=esta_bloqueado
        )
        
        if "Batch" in processing_mode:
            if proveedor == "claude":
                st.caption("📊 Track progress: https://console.anthropic.com/settings/workspaces/default/batches")
            else:
                st.caption("📊 Track progress: https://platform.openai.com/batches")
    else:
        processing_mode = "Normal (line by line)"

    # --- CONSTRUCCIÓN DEL PROMPT ---
    partes = [rol, contexto, clasificacion, formato, constraints, extra_content]
    prompt_final = "\n".join([p for p in partes if p.strip()])
    
    strategy_folder = estrategia.lower().replace(" ", "_") if estrategia else "default"

    run_label = st.text_input("Run label", value="run-1", disabled=esta_bloqueado,
                             help="Keep the same label to resume. Use a new label for an independent repetition.")
    run_id = run_fingerprint(info_llm["generation"], prompt_final,
                             dataset_digest(df), message_col, game_col, keep_cols, run_label)
    strategy_folder = os.path.join(strategy_folder, info_llm["modelo"], run_id)
    info_llm["run_id"] = run_id
    info_llm["run_folder"] = strategy_folder
    info_llm["prompt"] = prompt_final
    info_llm["run_label"] = run_label
    st.caption("Matching runs resume saved successful rows. Changes to data, prompt, model or reasoning start a separate run.")
    if st.button("Generate Prompt and Run", disabled=esta_bloqueado or not temps):
        if prompt_final.strip():
            st.session_state.proceso_finalizado = True
            st.session_state.stop_requested = False
            st.session_state.batch_status = False
            st.session_state["attempted_outputs"] = set()
            st.rerun()
        else:
            st.error("The prompt is empty.")


    # --- MOSTRAR RESULTADOS Y PROCESAMIENTO ---
    # si partes no esta empty y el proceso se ha marcado como finalizado, mostramos el prompt y ejecutamos
    if st.session_state.proceso_finalizado:
        # Opcional: Botón para resetear y volver a configurar
        if st.button("STOP, reset, and Edit Prompt", type="primary"):
            st.session_state.proceso_finalizado = False
            st.session_state.stop_requested= True
            st.session_state.batch_status = False
            st.rerun()
          
        st.subheader("Final Generated Prompt")
        st.code(prompt_final, language="markdown")
        
        # Llamamos a la función de procesamiento. 
        # Al estar fuera del 'if button', persistirá aunque hagas clic en descargar.
        if estrategia:
          
            if "Batch" in processing_mode: 
                
                ejecutar_batch_st(
                    df,
                    prompt_final,
                    info_llm,
                    temps,
                    message_col,
                    strategy_folder,
                    API_KEY,
                    game_col=game_col,
                    keep_cols=keep_cols,
                )


            else:
                
                st.caption("Processing row by row. To stop, click ** STOP, reset, and Edit Prompt ** — it will halt after the current API call finishes.")

                ejecutar_procesamiento_st(
                    df,
                    prompt_final,
                    info_llm,
                    temps,
                    message_col,
                    strategy_folder,
                    API_KEY,
                    game_col=game_col,
                    keep_cols=keep_cols,
                )



        else:
            
            ejecutar_procesamiento_crear_st(
                df, 
                prompt_final, 
                info_llm, 
                temps, 
                message_col,  
                API_KEY
            )
            


        if st.button("Start New Process", type="primary"):
            st.session_state.proceso_finalizado = False
            st.session_state.batch_status = False
            st.session_state.stop_requested= True

            st.rerun()

def configuracion_temperaturas(esta_bloqueado, config):
    if not config.allows_temperature:
        st.caption("Temperature is omitted for this model and reasoning configuration.")
        return [None]
    choices = [0.0, 0.1, 0.5, 1.0] if config.spec.provider == "claude" else [0.0, 0.1, 0.5, 1.0, 1.2]
    return st.multiselect("Temperatures", choices, default=[0.0], key=f"temps_{config.model}",
                          disabled=esta_bloqueado, help="Each temperature creates a separate result file.")


def save_run_manifest(config_llm, temp, output_path):
    metadata = config_llm["generation"].metadata(temp)
    metadata.update(run_id=config_llm.get("run_id"), prompt=config_llm.get("prompt", ""),
                    run_label=config_llm.get("run_label", "run-1"))
    path = Path(output_path).with_suffix(".metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(metadata, ensure_ascii=False, indent=2)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def dataset_digest(df):
    """Order-, column- and dtype-sensitive digest; far cheaper than serializing the frame on every rerun."""
    digest = hashlib.sha256(json.dumps([[str(c) for c in df.columns], [str(t) for t in df.dtypes]]).encode())
    try:
        digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    except TypeError:
        digest.update(df.to_json(orient="split", date_format="iso").encode())
    return digest.hexdigest()


class CacheUsage:
    """Prompt-cache token counts for one run; the lock makes it safe across worker threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self.input = self.cached = self.written = 0

    def record(self, llm, response):
        try:
            if llm == "claude":
                u = response.usage
                cached, written = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
                total = (u.input_tokens or 0) + cached + written
            elif llm == "chatgpt":
                u = response.usage
                total, written = u.prompt_tokens or 0, 0
                cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
            else:
                u = response.usage_metadata
                total, written = u.prompt_token_count or 0, 0
                cached = u.cached_content_token_count or 0
        except (AttributeError, TypeError):
            return
        with self._lock:
            self.input += total
            self.cached += cached
            self.written += written

    def summary(self):
        if not self.input:
            return ""
        text = f"Prompt cache: {self.cached:,} of {self.input:,} input tokens were read from cache"
        return text + (f"; {self.written:,} written to cache." if self.written else ".")


def temperature_label(temp):
    return "model default" if temp is None else str(temp)


def ejecutar_procesamiento_st(df, prompt, config_llm, temps, message_col, strategy_folder, API_KEY, game_col=None, keep_cols=None):
    
    df_clean = df.dropna(subset=[message_col]).reset_index(drop=True)
    
    proveedor = config_llm['proveedor']
    
    model_override = config_llm['modelo']
    
    keep_cols = keep_cols or []

    def enrich_row(parsed, idx, msg):
        parsed["row_id"] = idx
        parsed["original_message"] = msg
        for col in keep_cols:
            if col in df_clean.columns:
                parsed[col] = df_clean.at[idx, col]
        return parsed
        
    for temp in temps:
        
        
        if st.session_state.get('stop_requested'):
            break

        with st.container(border=True):
            st.subheader(f"Temperature: {temp}")

            bar = st.progress(0)
            status = st.empty()
            log_area = st.expander(f"See error logs (Temp {temp})")

            out_file = f"results_line_temp{temp}.csv"
            output_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, out_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_run_manifest(config_llm, temp, output_path)

            processed_ids = set()
            rows_buffer = []

            if os.path.exists(output_path):
                try:
                    existing_df = pd.read_csv(output_path, encoding='utf-8')
                    if "row_id" in existing_df.columns:
                        processed_ids = set(existing_df["row_id"].tolist())
                        rows_buffer = existing_df.to_dict('records')
                        status.info(f"Previous progress: {len(processed_ids)} detected lines.")
                except Exception as e:
                    status.warning(f"Could not read existing file (will restart): {e}")
                    processed_ids = set()
                    rows_buffer = []

            pending_rows = [
                (idx, row[message_col], row.get(game_col, None) if game_col else None)
                for idx, row in df_clean.iterrows()
                if idx not in processed_ids
            ]

            attempted = st.session_state.setdefault("attempted_outputs", set())
            if output_path in attempted and pending_rows:
                status.warning("Some rows are unfinished. Start a new process with the same settings to retry them.")
                if rows_buffer:
                    mostrar_boton_descarga(pd.DataFrame(rows_buffer), temp, "results", output_path)
                continue

            total_total = len(df_clean)

            if not pending_rows and len(processed_ids) >= total_total:
                bar.progress(1.0)
                status.success(f"Temperature {temp} completed previously.")
            else:
                stopped_early = False

                usage = CacheUsage()
                try:
                    if proveedor == "gemini":
                        # No `with`: its exit would wait for every queued request (already paid for,
                        # then discarded) when Streamlit interrupts the script for a rerun.
                        executor = ThreadPoolExecutor(max_workers=5)
                        try:
                            futures = {
                                executor.submit(
                                    call_llm_for_message, prompt, msg, temp, "gemini", API_KEY,
                                    game, model_override, "user", config_llm["generation"], usage=usage
                                ): (idx, msg)
                                for idx, msg, game in pending_rows
                            }
                            for i, future in enumerate(as_completed(futures)):
                                if st.session_state.get('stop_requested'):
                                    stopped_early = True
                                    break
                                idx, msg = futures[future]
                                try:
                                    ans = future.result()
                                    parsed = parse_llm_dict(ans)
                                    rows_buffer.append(enrich_row(parsed, idx, msg))
                                    if len(rows_buffer) % 10 == 0:
                                        pd.DataFrame(rows_buffer).to_csv(output_path, index=False)
                                    completados = len(processed_ids) + i + 1
                                    bar.progress(completados / total_total)
                                    status.markdown(f"**Gemini:** `{completados}/{total_total}`")
                                except Exception as e:
                                    log_area.error(f"Row {idx}: {e}")
                        finally:
                            executor.shutdown(wait=False, cancel_futures=True)

                    elif proveedor == "claude":
                        for i, (idx, msg, game) in enumerate(pending_rows):
                            if st.session_state.get('stop_requested'):
                                stopped_early = True
                                break
                            try:
                                ans = call_llm_for_message(
                                    prompt, msg, temp, "claude", API_KEY, game, model_override, "user", config_llm["generation"],
                                    usage=usage
                                )
                                parsed = parse_llm_dict(ans)
                                rows_buffer.append(enrich_row(parsed, idx, msg))
                                if len(rows_buffer) % 5 == 0:
                                    pd.DataFrame(rows_buffer).to_csv(output_path, index=False)
                                completados = len(processed_ids) + i + 1
                                bar.progress(completados / total_total)
                                status.markdown(f"**Claude:** `{completados}/{total_total}`")
                            except Exception as e:
                                log_area.error(f"Row {idx}: {e}")

                    else:
                        for i, (idx, msg, game) in enumerate(pending_rows):
                            if st.session_state.get('stop_requested'):
                                stopped_early = True
                                break
                            try:
                                ans = call_llm_for_message(
                                    prompt, msg, temp, "chatgpt", API_KEY, game, model_override, "user", config_llm["generation"],
                                    usage=usage
                                )
                                parsed = parse_llm_dict(ans)
                                rows_buffer.append(enrich_row(parsed, idx, msg))
                                if len(rows_buffer) % 5 == 0:
                                    pd.DataFrame(rows_buffer).to_csv(output_path, index=False)
                                completados = len(processed_ids) + i + 1
                                bar.progress(completados / total_total)
                                status.markdown(f"**GPT:** `{completados}/{total_total}`")
                            except Exception as e:
                                log_area.error(f"Row {idx}: {e}")
                finally:
                    # Pure Python: it also runs when Streamlit interrupts the script for a rerun.
                    if rows_buffer:
                        pd.DataFrame(rows_buffer).sort_values("row_id").to_csv(output_path, index=False)
                attempted.add(output_path)
                if usage.summary():
                    st.caption(usage.summary())
                if len(rows_buffer) < total_total:
                    status.warning(f"Saved {len(rows_buffer)} of {total_total} rows. Missing rows remain available to retry.")

                if stopped_early:
                    status.warning(f"Stopped at row {len(rows_buffer)}. Progress saved.")
                    break

            if rows_buffer:
                # The sorted file was already written above; only the download needs the frame.
                df_results = pd.DataFrame(rows_buffer).sort_values("row_id")
                mostrar_boton_descarga(df_results, temp, "results", output_path)

def mostrar_boton_descarga(df_temp, temp, type, output_path=None):
    
    st.success(f"Results ready (temperature: {temperature_label(temp)}).")
    
    csv_temp = df_temp.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label=f"Download results ({temperature_label(temp)})",
        data=csv_temp,
        file_name=f"{type}_temp_{temp}.csv",
        mime="text/csv",
        key=f"dl_{type}_{output_path}_{temp}"
    )
    
    if output_path:
        metadata_path = Path(output_path).with_suffix(".metadata.json")
        if metadata_path.exists():
            st.download_button("Download run configuration", metadata_path.read_bytes(),
                               file_name="run_configuration.json", mime="application/json",
                               key=f"meta_{output_path}")
    st.divider()

def write_rows_to_csv(output_path, rows):
    if not rows:
        return

    frame = pd.DataFrame(rows)
    if os.path.exists(output_path):
        frame = pd.concat([pd.read_csv(output_path), frame], ignore_index=True)
    frame = frame.drop_duplicates(subset="row_id", keep="last").sort_values("row_id")
    temporary = str(output_path) + ".tmp"
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, output_path)

# Clients are cached per API key: one connection pool is reused across rows, and a
# different key never reuses another key's client. Never close them with `with`.
@lru_cache(maxsize=8)
def get_gemini_client(API_KEY):
    return genai.Client(api_key=API_KEY)


@lru_cache(maxsize=8)
def get_claude_client(API_KEY):
    return anthropic.Anthropic(api_key=API_KEY, max_retries=2, timeout=180.0)


def extract_claude_text(response):
    if hasattr(response, 'content') and response.content:
        text_blocks = [b.text for b in response.content if hasattr(b, 'type') and b.type == 'text']
        return "\n".join(text_blocks) if text_blocks else ""
    return ""

def build_classification_user_message(message, game=None):
    preamble = f"The actual game being played this period is: Game {game}\n\n" if game is not None else ""
    return (
        "Classify ONLY this message and return only a Python dictionary. "
        "Include any justification field required by the supplied format.\n\n" +
        preamble +
        f"Message:\n{message}"
    )

def call_llm_for_message(base_prompt, message, temp, llm, API_KEY, game=None, model_override=None, mode="user", generation=None,
                         cache_user_message=False, usage=None):
    model = model_override or models_for(llm)[0]
    spec = MODELS[model]
    config = generation or GenerationConfig(model, spec.default_effort, spec.thinking_modes[0] if spec.thinking_modes else "")
    if config.model != model or config.spec.provider != llm:
        raise ValueError("Model and provider configuration do not match.")
    user_prompt = build_classification_user_message(message, game) if mode == "user" else str(message)
    params = request_payload(config, base_prompt, user_prompt, temp, cache_user_message=cache_user_message)
    if llm == "gemini":
        client = get_gemini_client(API_KEY)  # keep a reference: the SDK client closes itself when collected
        response = client.models.generate_content(**params)
        if usage:
            usage.record(llm, response)
        if response.candidates and str(response.candidates[0].finish_reason).endswith("MAX_TOKENS"):
            raise ValueError("Output token limit reached. Increase the limit or lower reasoning effort.")
        answer = response.text
    elif llm == "claude":
        # Streaming avoids the SDK's non-streaming timeout limit on larger budgets.
        with get_claude_client(API_KEY).messages.stream(**params) as stream:
            response = stream.get_final_message()
        if usage:
            usage.record(llm, response)
        if response.stop_reason == "max_tokens":
            raise ValueError("Output token limit reached. Increase the limit or lower reasoning effort.")
        answer = extract_claude_text(response)
    else:
        response = get_openai_batch_client(API_KEY).chat.completions.create(**params)
        if usage:
            usage.record(llm, response)
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise ValueError("No complete answer. Check the output token limit or provider refusal.")
        answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError("The model returned no text. Check reasoning/output limits or provider safety blocks.")
    return answer


def parse_llm_dict(ans):
    if not isinstance(ans, str):
        raise ValueError("The model returned no text.")
    start, end = ans.find("{"), ans.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("The model did not return a dictionary. Review the output format.")
    raw = ans[start:end]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Invalid dictionary returned; this row can be retried.") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("Expected a non-empty dictionary.")
    return parsed


def ejecutar_procesamiento_crear_st(df, prompt, config_llm, temps, message_col, API_KEY):    
    # Limpiamos el dataframe
    df_clean = df.dropna(subset=[message_col]).reset_index(drop=True)
    
    # Preparamos el bloque de mensajes una sola vez
    messages = df_clean[message_col].tolist()
    combined_messages = "\n".join([f"- {m}" for m in messages])
    
    combined_messages = "Messages to analyze to create categories:\n" + combined_messages
    
    for temp in temps:
        with st.container(border=True):
            st.subheader(f"Results with temperature: {temp}")
            
            proveedor = config_llm['proveedor']
            category_path = os.path.join(RESULTS_PATH, proveedor, config_llm["run_folder"], f"categories_temp{temp}.csv")
            with st.spinner(f"The LLM is analyzing and categorizing (Temp {temp})..."):
                try:
                    # 1. Llamada única al LLM
                    if os.path.exists(category_path):
                        res_df = pd.read_csv(category_path)
                        st.table(res_df)
                        mostrar_boton_descarga(res_df, temp, "categories", category_path)
                        continue
                    attempted = st.session_state.setdefault("attempted_outputs", set())
                    if category_path in attempted:
                        st.warning("This request failed. Start a new process to retry.")
                        continue
                    # A second breakpoint on the (large) messages only pays off when several
                    # temperatures of this run send the same messages.
                    ans = call_llm_for_message(prompt, combined_messages, temp, proveedor, API_KEY,
                                               model_override=config_llm["modelo"], mode="categories",
                                               generation=config_llm["generation"],
                                               cache_user_message=len(temps) > 1)


                    # 2. Parsear el resultado (JSON -> Dict/List)
                    categorias_data = parse_llm_dict(ans)
                    
                    # 3. Mostrar de forma "bonita"
                    if categorias_data:
                        res_df = None
    
                        if isinstance(categorias_data, dict):
                            # REVISIÓN PARA TU CASO ESPECÍFICO:
                            # Convertimos el dict {Llave: Valor} en una lista de tuplas [(Llave, Valor)]
                            # y le ponemos nombres de columnas claros.
                            res_df = pd.DataFrame(
                                list(categorias_data.items()), 
                                columns=["Category", "Description"]
                            )

                        # RENDERIZADO FINAL
                        if res_df is not None:
                            # Save before any st.* call: a rerun raised there would otherwise
                            # discard the paid answer and repeat the request.
                            save_run_manifest(config_llm, temp, category_path)
                            res_df.to_csv(category_path, index=False)
                            # Esto imprimirá la tabla estética en vez del JSON
                            st.table(res_df)

                            # Botón de descarga usando tu función
                            mostrar_boton_descarga(res_df, temp, "categories", category_path)
                        else:
                            st.warning("No se pudo estructurar la tabla.")
                    else:
                        st.warning("The LLM did not return any data or it could not be parsed.")

                except Exception as e:
                    # Only a failed request is remembered; a Streamlit rerun interrupt is a
                    # BaseException, so an interrupted request is simply retried.
                    st.session_state.setdefault("attempted_outputs", set()).add(category_path)
                    st.error(f"Error processing temp {temp}: {e}")
   
# ──────────────────────────────────────────────────────────────
# BATCH HELPERS
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=8)
def get_openai_batch_client(API_KEY):
    return OpenAI(api_key=API_KEY, max_retries=2, timeout=180.0)


def temp_to_id_token(temp):
    return "default" if temp is None else str(temp).replace(".", "p").replace("-", "m")


def get_openai_response_text(response_body):
    choices = response_body.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)

def get_openai_file_text(file_response):
    text_attr = getattr(file_response, "text", None)
    if callable(text_attr):
        return text_attr()
    if isinstance(text_attr, str):
        return text_attr
    if hasattr(file_response, "read"):
        data = file_response.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return str(file_response)

def _st_batch_status_path(provider):
    path = os.path.join(RESULTS_PATH, provider, "batch_status_st.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def load_st_batch_status(provider):
    p = _st_batch_status_path(provider)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError) as exc:
            raise RuntimeError("Cannot read batch tracking; restore this file before submitting more batches.") from exc
    return {}

def save_st_batch_status(provider, data):
    path = _st_batch_status_path(provider)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(path + ".tmp", path)

def _rowmap_path(provider, batch_id):
    return os.path.join(RESULTS_PATH, provider, f"rowmap_{batch_id}.json")

# ──────────────────────────────────────────────────────────────
# BATCH SUBMISSION (Streamlit)
# ──────────────────────────────────────────────────────────────

def _to_native(v):
    if hasattr(v, "item"):
        return v.item()
    return v

def ejecutar_batch_st(df, prompt, config_llm, temps, message_col, strategy_folder, API_KEY, game_col=None, keep_cols=None):
    proveedor = config_llm['proveedor']
    model_override = config_llm['modelo']
    keep_cols = keep_cols or []
    df_clean = df.dropna(subset=[message_col]).reset_index(drop=True)
    submitted = []
    failed = False

    # --- INICIO DEL SPINNER / STATUS ---
    with st.status("🚀 Preparing and submitting batches...", expanded=True) as status:
        if st.session_state.batch_status == False:
            for temp in temps:
                st.write(f"Processing temperature {temp}...")
                
                out_file = f"results_batch_temp{temp}.csv"
                output_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, out_file)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_run_manifest(config_llm, temp, output_path)

                try:
                    previous = [meta for meta in load_st_batch_status(proveedor).values()
                                if meta.get("output_path") == output_path]
                except RuntimeError as exc:
                    failed = True
                    st.error(str(exc))
                    continue
                if previous:
                    st.info("A batch for this exact run is already pending. Collect it below before submitting again.")
                    continue
                processed_ids = set()
                if os.path.exists(output_path):
                    try:
                        ex = pd.read_csv(output_path)
                        if "row_id" in ex.columns:
                            processed_ids = set(int(float(v)) for v in ex["row_id"].dropna())
                    except Exception: pass

                pending_rows = [
                    (idx, row[message_col], row.get(game_col, None) if game_col else None)
                    for idx, row in df_clean.iterrows()
                    if idx not in processed_ids
                ]

                if not pending_rows:
                    st.write(f"✅ Temp {temp}: All rows already processed.")
                    mostrar_boton_descarga(pd.read_csv(output_path), temp, "batch_results", output_path)
                    continue

                temp_token = temp_to_id_token(temp)
                row_id_map = {}
                for i, (idx, msg, game) in enumerate(pending_rows):
                    entry = {
                        "idx": int(idx),
                        "message": str(msg),
                        "game": _to_native(game) if game is not None else None,
                        "extra": {},
                    }
                    for col in keep_cols:
                        if col in df_clean.columns:
                            entry["extra"][col] = _to_native(df_clean.at[idx, col])
                    row_id_map[f"temp{temp_token}_row{i}"] = entry

                try:
                    # --- ENVÍO A LAS APIs ---
                    if proveedor == "claude":
                        st.write(f"📤 Uploading batch to Anthropic (Temp {temp})...")
                        batch_requests = []
                        for custom_id, entry in row_id_map.items():
                            user_message = build_classification_user_message(entry["message"], entry["game"])
                            batch_requests.append({
                                "custom_id": custom_id,
                                "params": request_payload(config_llm["generation"], prompt, user_message, temp, batch=True),

                            })
                        client = get_claude_client(API_KEY)
                        batch_obj = client.messages.batches.create(requests=batch_requests)
                        batch_id = batch_obj.id

                    else: # chatgpt
                        st.write(f"📤 Uploading batch to OpenAI (Temp {temp})...")
                        client = get_openai_batch_client(API_KEY)
                        batch_lines = []
                        for custom_id, entry in row_id_map.items():
                            user_message = build_classification_user_message(entry["message"], entry["game"])
                            batch_lines.append(json.dumps({
                                "custom_id": custom_id,
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": request_payload(config_llm["generation"], prompt, user_message, temp),

                            }, ensure_ascii=False))

                        input_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, f"batch_input_temp{temp}.jsonl")
                        with open(input_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(batch_lines) + "\n")

                        with open(input_path, "rb") as fh:
                            input_file = client.files.create(file=fh, purpose="batch")
                        batch_obj = client.batches.create(
                            input_file_id=input_file.id,
                            endpoint="/v1/chat/completions",
                            completion_window="24h",
                        )
                        batch_id = batch_obj.id

                    # Persistencia
                    with open(_rowmap_path(proveedor, batch_id), "w", encoding="utf-8") as f:
                        json.dump(row_id_map, f, ensure_ascii=False, indent=2)

                    status_data = load_st_batch_status(proveedor)
                    status_data[batch_id] = {
                        "batch_id": batch_id, "provider": proveedor, "model": model_override,
                        "temperature": temp, "strategy": strategy_folder,
                        "generation": config_llm["generation"].metadata(temp),
                        "output_path": output_path, "total_rows": len(pending_rows),
                    }
                    save_st_batch_status(proveedor, status_data)
                    submitted.append({"temp": temp, "batch_id": batch_id, "rows": len(pending_rows)})

                except Exception as e:
                    failed = True
                    st.error(f"Temp {temp}: failed — {e}")

            # Actualizar el título del status al terminar
            status.update(label="Batch submission finished" if submitted else "No new batches submitted",
                          state="error" if failed else "complete", expanded=False)
            st.session_state.batch_status = True
        else:
            # A later rerun (refresh, download...) must not report a normal state as an error.
            status.update(label="Batch submission already handled", state="complete", expanded=False)

    # --- FEEDBACK FINAL ---
    if submitted:
        # Mostramos el éxito general
        st.success(f"✅ {len(submitted)} batch(es) submitted successfully!")
        
        # Creamos una pequeña lista con los detalles de lo que se acaba de enviar
        for s in submitted:
            st.info(f"**Temp {s['temp']}** | ID: `{s['batch_id']}` | {s['rows']} rows")

        # Mensajes de ayuda según el proveedor
        if proveedor == "claude":
            st.caption("📊 Track at: [Anthropic Console](https://console.anthropic.com/settings/workspaces/default/batches)")
        else:
            st.caption("📊 Track at: [OpenAI Dashboard](https://platform.openai.com/batches)")

        # OPCIONAL: Si quieres que se actualice la lista de abajo automáticamente 
        # sin perder este mensaje, puedes poner un botón de refresco:
        if st.button("🔄 Refresh List & View Batches"):
            st.session_state.proceso_finalizado = True
            st.session_state.batch_status = True
            st.rerun()
        

# ──────────────────────────────────────────────────────────────
# BATCH COLLECTION (Streamlit)
# ──────────────────────────────────────────────────────────────

def mostrar_estado_batches_st(API_KEY, selected_provider):
    all_pending = {}

    for provider in (selected_provider,) if selected_provider in ("claude", "chatgpt") else ():
        try:
            all_pending.update(load_st_batch_status(provider))
        except RuntimeError as exc:
            # Keep the rest of the page usable; submissions stay blocked until the file is restored.
            st.error(str(exc))

    for batch_id, meta in st.session_state.get("collected_batches", {}).items():
        if meta["provider"] == selected_provider and os.path.exists(meta["output_path"]):
            mostrar_boton_descarga(pd.read_csv(meta["output_path"]), meta["temperature"],
                                   f"collected_{batch_id}", meta["output_path"])
    
    if not all_pending:
        return

    st.divider()
    st.subheader("📦 Batches Pending", anchor="pendientes")
    
    
    # Usamos un loop más limpio
    for batch_id, meta in all_pending.items():
        with st.expander(f"🔹 {meta['provider'].upper()} - {meta['total_rows']} filas ({batch_id[:8]}...)", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            col1.caption("Configuration")
            col1.markdown(f"**Temp:** `{meta['temperature']}` | **Strategy:** `{meta['strategy']}`")
            
            col2.caption("Batch ID")
            col2.code(batch_id, language=None)
            
            if col3.button("Check & Collect", key=f"collect_{batch_id}", use_container_width=True):
                st.session_state.batch_status = True
                _collect_batch_results_st(batch_id, meta, API_KEY)
                                
def _collect_batch_results_st(batch_id, meta, API_KEY):
    provider = meta["provider"]
    output_path = meta["output_path"]

    rowmap_file = _rowmap_path(provider, batch_id)
    if not os.path.exists(rowmap_file):
        st.error(f"Row map file not found for batch {batch_id}. Cannot collect results.")
        return

    with open(rowmap_file, encoding="utf-8") as f:
        row_id_map = json.load(f)

    rows_buffer = []
    succeeded = 0
    errored = 0

    try:
        if provider == "claude":
            client = get_claude_client(API_KEY)
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                counts = batch.request_counts
                st.warning(
                    f"Batch not finished yet. Status: **{batch.processing_status}** | "
                    f"succeeded: {counts.succeeded} | processing: {counts.processing} | errored: {counts.errored}"
                )
                return

            for result in client.messages.batches.results(batch_id):
                if result.result.type != "succeeded":
                    errored += 1
                    continue
                entry = row_id_map.get(result.custom_id)
                if entry is None:
                    errored += 1
                    continue
                try:
                    if result.result.message.stop_reason == "max_tokens":
                        raise ValueError("Truncated answer")
                    parsed = parse_llm_dict(extract_claude_text(result.result.message))
                except ValueError:
                    errored += 1
                    continue
                parsed["row_id"] = entry["idx"]
                parsed["original_message"] = entry["message"]
                for col, val in entry.get("extra", {}).items():
                    parsed[col] = val
                rows_buffer.append(parsed)
                succeeded += 1

        else:  # chatgpt
            client = get_openai_batch_client(API_KEY)
            batch = client.batches.retrieve(batch_id)
            if batch.status not in ("completed", "failed", "expired", "cancelled"):
                counts = batch.request_counts
                st.warning(
                    f"Batch not finished yet. Status: **{batch.status}** | "
                    f"completed: {counts.completed} | failed: {counts.failed} | total: {counts.total}"
                )
                return

            if not batch.output_file_id:
                st.error(f"Batch ended with status `{batch.status}` and no output file.")
                _remove_st_batch(provider, batch_id)
                return

            output_text = get_openai_file_text(client.files.content(batch.output_file_id))
            for line in output_text.splitlines():
                if not line.strip():
                    continue
                result = json.loads(line)
                entry = row_id_map.get(result.get("custom_id"))
                if entry is None:
                    errored += 1
                    continue
                if result.get("error") or result.get("response", {}).get("status_code") != 200:
                    errored += 1
                    continue
                body = result.get("response", {}).get("body", {})
                try:
                    if not body.get("choices") or body["choices"][0].get("finish_reason") != "stop":
                        raise ValueError("Incomplete answer")
                    parsed = parse_llm_dict(get_openai_response_text(body))
                except ValueError:
                    errored += 1
                    continue
                parsed["row_id"] = entry["idx"]
                parsed["original_message"] = entry["message"]
                for col, val in entry.get("extra", {}).items():
                    parsed[col] = val
                rows_buffer.append(parsed)
                succeeded += 1

    except Exception as e:
        st.error(f"Error collecting batch results: {e}")
        return

# Al final, cuando ya tenemos los datos, presentamos el resultado bonito:
    if rows_buffer:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_rows_to_csv(output_path, rows_buffer)
        
        # UI limpia de éxito
        st.toast("Results collected and saved!", icon="✅")
        
        with st.container(border=True):
            
            c1, c2 = st.columns(2)
            c1.metric("Successful Rows", succeeded)
            c2.metric("Errors", errored, delta=errored, delta_color="inverse")
            
            # Previsualización opcional en un pequeño dataframe

            
            # Botón de descarga destacado
            df_results = pd.read_csv(output_path)
            mostrar_boton_descarga(df_results, meta["temperature"], "batch_results", output_path)
            st.session_state.setdefault("collected_batches", {})[batch_id] = meta
    else:
        st.warning("No valid rows were returned. Review the prompt and token limit before retrying.")
    if errored:
        st.warning(f"{errored} returned rows failed or were invalid. Only successful rows are saved; retry the run to process missing rows.")
    # The batch is terminal. Archive its mapping so a new submission can retry missing rows.
    archive = Path(rowmap_file).with_name(f"completed_{batch_id}.json")
    os.replace(rowmap_file, archive)
    _remove_st_batch(meta["provider"], batch_id)
        
def _remove_st_batch(provider, batch_id):
    status = load_st_batch_status(provider)
    if batch_id in status:
        del status[batch_id]
        save_st_batch_status(provider, status)
    rowmap = _rowmap_path(provider, batch_id)
    if os.path.exists(rowmap):
        os.remove(rowmap)

# ──────────────────────────────────────────────────────────────
# MAIN STREAMLIT APP
# ──────────────────────────────────────────────────────────────

def main():

    st.title("Responses classification with LLMs")
    st.caption(f"Version {APP_VERSION}")

    # Initialize session state flags
    if 'proceso_finalizado' not in st.session_state:
        st.session_state.proceso_finalizado = False
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False
    if 'batch_status' not in st.session_state:
        st.session_state.batch_status = False
        
    if st.button("Reset / Clear Screen"):
        # Reset only the UI; keep results and batch tracking on disk.

        st.session_state.proceso_finalizado = False
        st.session_state.stop_requested = False
        st.session_state.batch_status = False
        
        st.rerun()

    # 1. El usuario sube el archivo
    archivo_subido = st.file_uploader("Upload your CSV file", type="csv", disabled=st.session_state.proceso_finalizado)

    if archivo_subido is not None:
        # Cargamos el dataframe para que esté disponible en las acciones
        try:
            df = pd.read_csv(archivo_subido)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            return
        if df.empty:
            st.warning("Upload a CSV with at least one data row.")
            return
        st.success("File successfully uploaded")
        
        st.write("### Preliminary data view:")
        st.dataframe(df.head())

        # 2. Preguntar qué se quiere hacer (reemplaza tus menús anteriores)
        accion = st.segmented_control(
            "What would you like to do?", 
            options=["Create categories", "Assign categories"], 
            selection_mode="single", 
            disabled=st.session_state.proceso_finalizado
        )

        # 3. Lógica de ejecución según la acción seleccionada
        if accion == "Create categories":
            st.subheader("Configuration of new categories")
            #primero obtener categorias, depues hacer el mismo proceso
            menu_st(df, 1)

        elif accion == "Assign categories":

            st.subheader("Assign categories to csv")
            
            menu_st(df, 2)


    else:
        st.info("Waiting for CSV to abilitate actions")
        
if __name__ == "__main__":   
    main()
