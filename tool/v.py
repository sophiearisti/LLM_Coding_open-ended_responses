import streamlit as st
import pandas as pd
import ast
import re
import json
from http import client
from langchain_openai import ChatOpenAI
from google import genai
from google.genai import types
from tqdm import tqdm
from utils import *
import pandas as pd
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import anthropic
from openai import OpenAI


llm_chatgpt = None
llm_openai_batch = None
llm_gemini = None
llm_claude = None

GEMINI_CATEGORY_MODEL = "gemini-3.1-pro-preview"
GEMINI_CLASSIFY_MODEL = "gemini-3-flash-preview"
CLAUDE_CATEGORY_MODEL = "claude-opus-4-6"
GEMINI_WORKERS = 20
CLAUDE_MAX_TOKENS = 4096

# Selected models (updated at runtime via seleccionar_llm)
SELECTED_CHATGPT_MODEL = "gpt-5.4-mini"
SELECTED_GEMINI_MODEL = "gemini-3-flash-preview"
SELECTED_CLAUDE_MODEL = "claude-sonnet-4-6"
MIN_APPEND_KEY_OVERLAP = 0.90

def seleccionar_llm_st():
    st.markdown("### Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        proveedor = st.selectbox("Provider", ["ChatGPT", "Gemini", "Claude"], index=0)

    API_KEY = st.text_input("Enter your API Key", type="default")
    
    with col2:
        if proveedor == "ChatGPT":
            modelos = gpt_models
            # El recomendado suele ser el primero o uno específico
            modelo_elegido = st.selectbox("ChatGPT Model", modelos, index=0)
            return {"proveedor": "chatgpt", "modelo": modelo_elegido}, API_KEY
            
        elif proveedor == "Gemini":
            modelos = gemini_models
            modelo_elegido = st.selectbox("Gemini Model", modelos, index=0)
            return {"proveedor": "gemini", "modelo": modelo_elegido}, API_KEY
        
        elif proveedor == "Claude":
            modelos = claude_models
            modelo_elegido = st.selectbox("Claude Model", modelos, index=0)
            return {"proveedor": "claude", "modelo": modelo_elegido}, API_KEY

    return None

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
                return file.read().decode("utf-8")
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
                return file.read().decode("utf-8")
        elif method == "Pre-structured JSON":
            # 1. Primer bloque de texto (Prefijo)
            prefix = st.text_area("Initial Context (Text):", key=f"prefix_{label}", height=100, disabled = esta_bloqueado)

            st.markdown("---")
            st.write("🔧 **Data Structure**")

            # 2. Inicialización con las 3 filas por defecto (A, B, C)
            key_rows = f"rows_{label}"
            
            if type == "create":
                st.session_state[key_rows] = [
                    {"key": "A", "value": "the category is A: <definition of the category>"},
                    {"key": "B", "value": "the category is B: <definition of the category>"},
                    {"key": "C", "value": "the category is C: <definition of the category>"}
                ]
            else:
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
            full_content = f"{prefix}\n{json_part}\n{suffix}"
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
            horizontal=True
        )
        
    elif menu_type == 1:
        st.markdown("---")
        st.subheader("Categories Creation Strategy")

    info_llm, API_KEY = seleccionar_llm_st()

    if info_llm and API_KEY:
        # Always show pending batches, regardless of whether a file is uploaded
        crear_prompt_obtener_resultados(info_llm, df, API_KEY, estrategia, menu_type)
        
        mostrar_estado_batches_st(API_KEY)
             
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

    st.info("Temperatures Configuration")
    temps = configuracion_temperaturas(esta_bloqueado)
    
    # --- PROCESSING MODE (only for Claude / GPT) ---
    proveedor = info_llm['proveedor']
    
    temps = normalize_temps_for_llm(temps, proveedor)
    
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

    # --- VALIDACIÓN DE ARCHIVOS PREVIOS ---

    modos_ejecucion = {}

    # 1. Inicializamos o limpiamos los acumuladores en el estado de la sesión antes del bucle
    st.session_state["batches_a_eliminar_acumulados"] = []
    
    # Banderas globales para controlar el checkbox de confirmación
    al_menos_un_archivo_previo = False
    al_menos_un_json_previo = False

    for temp in temps:
        if esta_bloqueado == False:
            if "Batch" in processing_mode:
                
                hay_archivos_previos, modos_ejecucion = verify_files_existence(proveedor, strategy_folder, temp, modos_ejecucion, batch_prefix="batch", esta_bloqueado=esta_bloqueado)
                if hay_archivos_previos:
                    al_menos_un_archivo_previo = True

                # Ejecutamos la verificación del JSON
                hay_archivos_previos_json, batches_temp = verify_files_existence_json(df, proveedor, message_col, esta_bloqueado, temp)
                if hay_archivos_previos_json:
                    al_menos_un_json_previo = True
                
                # 🔴 ACUMULAMOS en lugar de sobrescribir
                if batches_temp:
                    st.session_state["batches_a_eliminar_acumulados"].extend(batches_temp)
                    
                print(f"Modos ejecución (Temp {temp}):", modos_ejecucion)
                print(f"Batches detectados en esta temp:", batches_temp)
            
            else:
                hay_archivos_previos, modos_ejecucion = verify_files_existence(proveedor, strategy_folder, temp, modos_ejecucion, batch_prefix="line", esta_bloqueado=esta_bloqueado)
                if hay_archivos_previos:
                    al_menos_un_archivo_previo = True
                    
                hay_archivos_previos_json = False
        else:
            # Si está bloqueado, no acumulamos nada
            modos_ejecucion = {}
    
    # --- VALIDACIÓN FINAL PARA MOSTRAR EL BOTÓN ---
    # Usamos las variables globales que no se sobrescriben destructivamente
    confirmado = True
    if al_menos_un_archivo_previo or al_menos_un_json_previo:
        confirmado = st.checkbox("I confirm the file management actions above.", value=False, disabled=esta_bloqueado)

    # --- BOTÓN DE EJECUCIÓN (Solo si está confirmado) ---
    if confirmado:
        if st.button("Generate Prompt and Run", disabled=esta_bloqueado):
            if prompt_final.strip():
                print("Modos de ejecución definidos para borrar CSVs:", modos_ejecucion)
                
                # 1. Limpieza de archivos CSV configurados en 'overwrite' y sus JSONL asociados
                if modos_ejecucion:
                    for t, info_modo in modos_ejecucion.items():
                        if info_modo.get('modo') == "overwrite":
                            output_path = info_modo.get('path') # 'Results/chatgpt/zero-shot/results_batch_temp0.5.csv'
                            
                            if output_path:
                                # Extrae dinámicamente 'Results/chatgpt/zero-shot/'
                                base_dir = os.path.dirname(output_path)
                                
                                # Construye la ruta al archivo .jsonl usando la temperatura t actual
                                jsonl_file = f"batch_input_temp{t}.jsonl"
                                jsonl_path = os.path.join(base_dir, jsonl_file)
                                
                                if os.path.exists(jsonl_path):
                                    os.remove(jsonl_path)
                                    print(f"JSONL eliminado: {jsonl_path}")
                                
                               
                                if os.path.exists(output_path):
                                    os.remove(output_path)
                                    print(f"CSV eliminado: {output_path}")
                                
                batches_para_borrar = st.session_state.get("batches_a_eliminar_acumulados", [])
                print("Batches a eliminar definitivos desde el State:", batches_para_borrar)  
                             
                if batches_para_borrar:
                    for item in batches_para_borrar:
                        batch_id = item["id"]
                        json_path = item["file"]
                        
                        # Eliminar el archivo rowmap del disco
                        if os.path.exists(json_path):
                            os.remove(json_path)
                            print(f"JSON eliminado: {json_path}")
                            
                        # Eliminar la entrada del JSON de estados (memoria de Streamlit)
                        _remove_st_batch(proveedor, batch_id)
                    
                    # Limpiamos el estado tras la eliminación
                    st.session_state["batches_a_eliminar_acumulados"] = []
                                
                st.session_state.proceso_finalizado = True
                st.session_state.stop_requested = False 
                st.rerun()
            else:
                st.error("The prompt is empty.")
    else:
        st.info("Please confirm the file management actions to proceed.")
        
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

def verify_files_existence_json(df, provider, message_col, esta_bloqueado, temp):
    MIN_APPEND_KEY_OVERLAP = 0.1
    # Usamos un set para evitar duplicados por rutas redundantes
    json_files = set()
    batches_a_eliminar = [] 
    
    provider_path = os.path.join(RESULTS_PATH, provider)
    
    if not os.path.exists(provider_path):
        return False, []

    for root, dirs, files in os.walk(provider_path):
        for file in files:
            if file.startswith("rowmap_") and file.endswith(".json"):
                # .add() en lugar de .append()
                json_files.add(os.path.join(root, file))
                
    # ==================== TU NUEVA LÓGICA DE FILTRADO POR TEMP ====================
    # Ruta del archivo de estados de los lotes (ajústala si está en otra subcarpeta)
    batch_status_path = os.path.join(RESULTS_PATH, provider,"batch_status_st.json")
    
    if os.path.exists(batch_status_path) and json_files:
        try:
            with open(batch_status_path, "r", encoding="utf-8") as f:
                batch_status_data = json.load(f)
                print(f"Loaded batch status data: {batch_status_data}")
            
            invalidos = set()
            for json_file in json_files:
                # Extraer el nombre base del archivo, ej: rowmap_batch_6a0a6d8757d4...json
                base_name = os.path.basename(json_file)
                
                # Reemplazamos 'rowmap_' por '' y quitamos '.json' para obtener 'batch_6a0a6d8757d4...'
                batch_code = base_name.replace("rowmap_", "").replace(".json", "")
                
                print(f"Checking batch code: {batch_code} against temperature {temp}")
                
                # Buscamos en el diccionario de estados si existe el lote y validamos su temperatura
                if batch_code in batch_status_data:
                    # Forzamos la comparación a float o int según corresponda para evitar errores de tipo
                    if float(batch_status_data[batch_code].get("temperature", -1)) != float(temp):
                        invalidos.add(json_file)
                else:
                    # Si por alguna razón el lote no está en el registro, podrías decidir descartarlo
                    invalidos.add(json_file)
                                
            # Removemos los que no corresponden a la temperatura seleccionada
            json_files.difference_update(invalidos)
            
        except Exception as e:
            st.error(f"Error filtering files by temperature: {e}")
    # ==============================================================================

    mensajes_df = set(df[message_col].dropna().astype(str).tolist())
    if not mensajes_df:
        return False, []
    
    hay_solapamiento = False
    

    # El bucle funciona igual sobre el set disminuido
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            mensajes_json = set(str(obj.get("message", "")) for obj in data.values())
            interseccion = mensajes_df.intersection(mensajes_json)
            overlap = len(interseccion) / len(mensajes_df) if len(mensajes_df) > 0 else 0
            
            if overlap >= MIN_APPEND_KEY_OVERLAP:
                # Extraer batch_id
                match = re.search(r"rowmap_(.*?)\.json", os.path.basename(json_file))
                if not match: continue
                batch_id = match.group(1)

                hay_solapamiento = True
                
                st.warning(f"⚠️ Overlap detected ({int(overlap*100)}%) in **{provider.upper()}** with Batch: `{batch_id}`")
                
                opcion = st.radio(
                    f"Action for batch {batch_id}:",
                    ["Maintain (Wait for it)", "Delete and Start Fresh"],
                    key=f"radio_json_{batch_id}",
                    disabled = esta_bloqueado
                )
                
                if "Delete" in opcion:
                    batches_a_eliminar.append({
                        "id": batch_id,
                        "file": json_file
                    })
                else:
                    st.info(f"Batch `{batch_id}` will be kept.")

        except Exception as e:
            st.error(f"Error checking {json_file}: {e}")
            continue

    print(f"batches a eliminar: {batches_a_eliminar}")
    
    return hay_solapamiento, batches_a_eliminar

def verify_files_existence(proveedor, strategy_folder, temp, modos_ejecucion, batch_prefix, esta_bloqueado):
    # Definimos el nombre del archivo específico para este prefijo y temperatura
    out_file = f"results_{batch_prefix}_temp{temp}.csv"
    output_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, out_file)
        
    hay_archivos_previos = False
    
    if os.path.exists(output_path):
        hay_archivos_previos = True
        st.warning(f"⚠️ Previous results detected for Temp {temp} (File: {out_file})")
        
        opcion = st.radio(
            f"Action for {out_file}:",
            ["Maintain (Append)", "Delete and Start Fresh (Overwrite)"],
            key=f"radio_{batch_prefix}_{strategy_folder}_{temp}"
        )
        
        # Guardamos un diccionario con toda la info necesaria
        modos_ejecucion[temp] = {
            "modo": "overwrite" if "Delete" in opcion else "append",
            "prefix": batch_prefix,
            "path": output_path
        }
    else:
        # Si es nuevo, también guardamos el prefijo para saber cómo crearlo
        modos_ejecucion[temp] = {
            "modo": "new",
            "prefix": batch_prefix,
            "path": output_path
        }
         
    return hay_archivos_previos, modos_ejecucion
             
def configuracion_temperaturas(esta_bloqueado):
    
    # Sin crear columnas ni usar 'with'
    temps = st.multiselect("Temperatures", [0, 0.1, 0.5, 1, 1.2], default=[0], help="Select the temperatures to run. You can choose multiple to compare results.", disabled=esta_bloqueado)
    
    return temps

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

            total_total = len(df_clean)

            if not pending_rows and len(processed_ids) >= total_total:
                bar.progress(1.0)
                status.success(f"Temperature {temp} completed previously.")
            else:
                stopped_early = False

                if proveedor == "gemini":
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = {
                            executor.submit(
                                call_llm_for_message, prompt, msg, temp, "gemini", API_KEY,
                                game, model_override, "user"
                            ): (idx, msg)
                            for idx, msg, game in pending_rows
                        }
                        for i, future in enumerate(as_completed(futures)):
                            if st.session_state.get('stop_requested'):
                                executor.shutdown(wait=False, cancel_futures=True)
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

                elif proveedor == "claude":
                    for i, (idx, msg, game) in enumerate(pending_rows):
                        if st.session_state.get('stop_requested'):
                            stopped_early = True
                            break
                        try:
                            ans = call_llm_for_message(
                                prompt, msg, temp, "claude", "user", game, model_override, API_KEY, "user"
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
                                prompt, msg, temp, "chatgpt", API_KEY, game, model_override, "user"
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

                pd.DataFrame(rows_buffer).to_csv(output_path, index=False)

                if stopped_early:
                    status.warning(f"Stopped at row {len(rows_buffer)}. Progress saved.")
                    break

            if rows_buffer:
                df_clean = pd.DataFrame(rows_buffer)
                df_clean.to_csv(output_path, index=False)
                # Convertimos la lista a DataFrame aquí
                mostrar_boton_descarga(df_clean, temp, "results")

def mostrar_boton_descarga(df_temp, temp, type):
    
    st.success(f"✅ ¡Temp {temp} ready to be downloaded!")
    
    csv_temp = df_temp.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label=f" Download results Temp {temp}",
        data=csv_temp,
        file_name=f"{type}_temp_{temp}.csv",
        mime="text/csv",
        key=f"dl_{temp}"
    )
    
    st.divider()
           
def write_rows_to_csv(output_path, rows):
    if not rows:
        return

    file_exists = os.path.exists(output_path)
    pd.DataFrame(rows).to_csv(
        output_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )

def get_chatgpt_client(API_KEY):
    global llm_chatgpt
    if llm_chatgpt is None or llm_chatgpt.model_name != SELECTED_CHATGPT_MODEL:
        llm_chatgpt = ChatOpenAI(
            model=SELECTED_CHATGPT_MODEL,
            max_retries=1,
            api_key=API_KEY
        )
    return llm_chatgpt

def get_gemini_client(API_KEY):
    global llm_gemini
    if llm_gemini is None:
        llm_gemini = genai.Client(api_key=API_KEY)
    return llm_gemini

def get_claude_client(API_KEY):
    global llm_claude
    if llm_claude is None:
        if anthropic is None:
            raise ImportError("Anthropic SDK is not installed. Run 'pip install anthropic'.")
        llm_claude = anthropic.Anthropic(api_key=API_KEY)
    return llm_claude

def extract_claude_text(response):
    if hasattr(response, 'content') and response.content:
        text_blocks = [b.text for b in response.content if hasattr(b, 'type') and b.type == 'text']
        return "\n".join(text_blocks) if text_blocks else ""
    return ""

def build_classification_user_message(message, game=None):
    preamble = f"The actual game being played this period is: Game {game}\n\n" if game is not None else ""
    return (
        "Classify ONLY this message and return only a Python dictionary. "
        "Do not add explanations.\n\n" +
        preamble +
        f"Message:\n{message}"
    )

def call_llm_for_message(base_prompt, message, temp, llm, API_KEY, game= None, model_override=None, mode="user"):
    
    if game and model_override:
        user_prompt = build_classification_user_message(message, game)
    else:
        user_prompt = ""

    if llm == "gemini":
        model = model_override or SELECTED_GEMINI_MODEL
        full_prompt = base_prompt + "\n\n" + user_prompt
        response = get_gemini_client(API_KEY).models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=temp)
        )
        return response.text

    if llm == "claude":
        model = model_override or SELECTED_CLAUDE_MODEL
        response = get_claude_client(API_KEY).messages.create(
            model=model,
            max_tokens=CLAUDE_MAX_TOKENS,
            temperature=temp,
            system=[{"type": "text", "text": base_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return extract_claude_text(response)

    model = model_override or SELECTED_CHATGPT_MODEL
    global llm_chatgpt
    if llm_chatgpt is None or llm_chatgpt.model_name != model:
        llm_chatgpt = get_chatgpt_client(API_KEY)

    if mode == "user":
        response = get_chatgpt_client(API_KEY).bind(temperature=temp).invoke([
            ("system", base_prompt),
            ("user", user_prompt)
        ])
    else:
        response = get_chatgpt_client(API_KEY).invoke_as_assistant(user_prompt, temperature=temp)

    return response.content

def parse_llm_dict(ans):
    try:
        start = ans.find("{")
        end   = ans.rfind("}") + 1
        raw_dict = ans[start:end]
        return ast.literal_eval(raw_dict)
    except:
        return {"error": ans[:300]}

def ejecutar_procesamiento_crear_st(df, prompt, config_llm, temps, message_col, API_KEY):    
    # Limpiamos el dataframe
    df_clean = df.dropna(subset=[message_col]).reset_index(drop=True)
    
    # Preparamos el bloque de mensajes una sola vez
    messages = df_clean[message_col].tolist()
    combined_messages = "\n".join([f"- {m}" for m in messages])
    
    full_prompt = (
        f"{prompt}\n\n"
        "These are the messages you need to analyze to create the categories:\n"
        f"{combined_messages}"
    )
    
    for temp in temps:
        with st.container(border=True):
            st.subheader(f"Results with temperature: {temp}")
            
            with st.spinner(f"The LLM is analyzing and categorizing (Temp {temp})..."):
                try:
                    # 1. Llamada única al LLM
                    proveedor = config_llm['proveedor']
                    
                    if proveedor == "gemini":
                        # Usamos tu función existente call_llm_for_message pero enviando el bloque completo
                        ans = call_llm_for_message(full_prompt, "", temp, "gemini", API_KEY)
                        
                    elif proveedor == "claude":
                        ans = call_llm_for_message(full_prompt, "", temp, "claude", API_KEY)
                    else:
                        ans = call_llm_for_message(full_prompt, "", temp, "chatgpt", API_KEY)

                    
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
                            # Esto imprimirá la tabla estética en vez del JSON
                            st.table(res_df) 
                            
                            # Botón de descarga usando tu función
                            mostrar_boton_descarga(res_df, temp, "categories")
                        else:
                            st.warning("No se pudo estructurar la tabla.")
                    else:
                        st.warning("The LLM did not return any data or it could not be parsed.")

                except Exception as e:
                    st.error(f"Error processing temp {temp}: {e}")    
   
# ──────────────────────────────────────────────────────────────
# BATCH HELPERS
# ──────────────────────────────────────────────────────────────
def get_openai_batch_client(API_KEY):
    global llm_openai_batch
    if llm_openai_batch is None:
        llm_openai_batch = OpenAI(api_key=API_KEY)
    return llm_openai_batch

def temp_to_id_token(temp):
    return str(temp).replace(".", "p").replace("-", "m")

def normalize_temps_for_llm(temps, llm):
    if llm != "claude":
        return temps
    valid = [t for t in temps if 0 <= float(t) <= 1]
    dropped = [t for t in temps if not (0 <= float(t) <= 1)]
    if dropped:
        st.warning(f"Claude only supports temperature 0–1. Skipping: {dropped}")
    return valid if valid else [0]

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
        except Exception:
            pass
    return {}

def save_st_batch_status(provider, data):
    with open(_st_batch_status_path(provider), "w") as f:
        json.dump(data, f, indent=2)

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

    # --- INICIO DEL SPINNER / STATUS ---
    with st.status("🚀 Preparing and submitting batches...", expanded=True) as status:
        if st.session_state.batch_status == False:
            for temp in temps:
                st.write(f"Processing temperature {temp}...")
                
                out_file = f"results_batch_temp{temp}.csv"
                output_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, out_file)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
                                "params": {
                                    "model": model_override,
                                    "max_tokens": CLAUDE_MAX_TOKENS,
                                    "temperature": temp,
                                    "system": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
                                    "messages": [{"role": "user", "content": user_message}],
                                },
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
                                "body": {
                                    "model": model_override,
                                    "temperature": temp,
                                    "messages": [
                                        {"role": "system", "content": prompt},
                                        {"role": "user", "content": user_message},
                                    ],
                                },
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
                        "output_path": output_path, "total_rows": len(pending_rows),
                    }
                    save_st_batch_status(proveedor, status_data)
                    submitted.append({"temp": temp, "batch_id": batch_id, "rows": len(pending_rows)})

                except Exception as e:
                    st.error(f"Temp {temp}: failed — {e}")

        # Actualizar el título del status al terminar
        status.update(label="✅ Batches Submitted Successfully!", state="complete", expanded=False)
        st.session_state.batch_status = True

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

def mostrar_estado_batches_st(API_KEY):
    all_pending = {}
    
    for provider in ("claude", "chatgpt"):
        for batch_id, meta in load_st_batch_status(provider).items():
            all_pending[batch_id] = meta
    
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
                parsed = parse_llm_dict(extract_claude_text(result.result.message))
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
                ans = get_openai_response_text(result.get("response", {}).get("body", {}))
                parsed = parse_llm_dict(ans)
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
            df_results = pd.DataFrame(rows_buffer)
            mostrar_boton_descarga(df_results, meta["temperature"], "batch_results")
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

    # Initialize session state flags
    if 'proceso_finalizado' not in st.session_state:
        st.session_state.proceso_finalizado = False
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False
    if 'batch_status' not in st.session_state:
        st.session_state.batch_status = False
        
    if st.button("Reset / Clear Screen"):
        folder = "Results"

        if os.path.exists(folder):
            shutil.rmtree(folder)

        os.makedirs(folder)
        
        st.session_state.proceso_finalizado = False
        st.session_state.stop_requested = False
        st.session_state.batch_status = False
        
        st.rerun()

    # 1. El usuario sube el archivo
    archivo_subido = st.file_uploader("Upload your CSV file", type="csv")

    if archivo_subido is not None:
        # Cargamos el dataframe para que esté disponible en las acciones
        df = pd.read_csv(archivo_subido)
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