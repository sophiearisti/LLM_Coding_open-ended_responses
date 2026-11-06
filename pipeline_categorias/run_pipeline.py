#!/usr/bin/env python3
"""Pipeline automático para respuestas abiertas:

  1) Claude Opus 5.5 lee TODAS las respuestas de cada pregunta y crea las categorías
     (sin límite de número).
  2) Claude Sonnet 5.5, en modo Batch (50 % más barato), clasifica cada respuesta con
     esas categorías.
  3) Todo se guarda en ./resultados/.

Es reanudable: si se interrumpe, vuelve a correr el mismo comando y continúa donde quedó
(nunca reenvía un lote ya enviado ni repite categorías ya creadas).

Uso rápido (ver README.md):
    python run_pipeline.py --preparar ARCHIVO_LIMPIO.csv CODEBOOK.csv   # una sola vez
    python run_pipeline.py --dry-run     # sin API, valida todo
    python run_pipeline.py --prueba      # unas pocas llamadas reales (cuesta centavos)
    python run_pipeline.py               # corrida completa
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN (edita aquí)
# ──────────────────────────────────────────────────────────────────────────────
PREGUNTAS = ["Q14", "Q15", "Q16", "Q17", "Q18", "Q21", "Q24", "Q27", "Q30"]
# Campos "Otro, especifique" (descomenta si los quieres incluir):
#   "Q09_97_TEXT", "Q09C_97_TEXT", "Q32_97_TEXT"

# Variables que se copian al archivo FINAL para poder cruzar con la base original y analizar por
# grupo (junto con ResponseId, que es la llave). NO se envían al modelo ni se usan para categorizar.
# Q02 género, Q03 edad, Q05 tipo de vinculación, Q04 departamento (solo profesores de planta),
# Q06 años de experiencia, Q07 programas donde enseñó. Excluimos a propósito IPAddress, correo y ubicación.
COLUMNAS_META = ["StartDate", "EndDate", "Progress", "Finished", "Q02", "Q03", "Q05", "Q04", "Q06", "Q07"]

# El contexto de la encuesta (descripción, situaciones/viñetas de Q18, Q21, Q24 y Q27, lógica de
# salto y notas por pregunta) vive en contexto_preguntas.json, tomado del cuestionario original.
# Edítalo ahí; se inyecta en los prompts de crear y de asignar categorías.
CONTEXTO_ENCUESTA_BASE = (
    "Encuesta diagnóstica a profesores universitarios sobre la integración de la "
    "IA generativa en su docencia (usos, evaluación, decisiones éticas y políticas)."
)

MODELO_CATEGORIAS = "claude-opus-5-5"
ESFUERZO_CATEGORIAS = "high"          # Opus 5.5 usa "medium" por defecto; lo fijamos explícito
MAX_TOKENS_CATEGORIAS = 64000         # el razonamiento cuenta dentro de este límite

MODELO_CLASIFICACION = "claude-sonnet-5-5"
ESFUERZO_CLASIFICACION = "medium"
MAX_TOKENS_CLASIFICACION = 8192

MULTI_ETIQUETA = True                 # True: una respuesta puede tener varias categorías
MAX_INTENTOS_POR_RESPUESTA = 2        # reintentos si una respuesta falla en el lote
MAX_RONDAS_LOTE = 3                   # máx. lotes consecutivos por corrida
SEGUNDOS_ENTRE_CONSULTAS = 60         # espera entre consultas de estado del lote

# ──────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESP_CSV = DATA / "respuestas.csv"
CODEBOOK_CSV = DATA / "codebook.csv"
META_CSV = DATA / "metadatos.csv"
CONTEXTO_JSON = BASE / "contexto_preguntas.json"
OUT = BASE / "resultados"
DIR_CAT = OUT / "categorias"
DIR_RAW = OUT / "clasificacion_raw"
DIR_CLA = OUT / "clasificacion"
ESTADO = OUT / "estado.json"
LOG = OUT / "pipeline.log"

PLACEHOLDERS = {"na", "n/a", "nan", "null", "none", "ns/nr"}
RESERVADOS = {"texto", "responseid", "estado_clasificacion"} | {c.lower() for c in COLUMNAS_META}  # columnas de salida
log = logging.getLogger("pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
def ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def guardar_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def leer_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def es_respuesta(texto):
    """False para vacíos y relleno tipo '-', '.', 'NA'."""
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return False
    s = str(texto).strip()
    return len(re.sub(r"[\W_]+", "", s)) >= 2 and s.lower() not in PLACEHOLDERS


def linea(valor, maximo):
    """Una sola línea, sin saltos ni caracteres de control, con largo máximo.
    Se usa en lo que genera el modelo (categorías) antes de ponerlo en un prompt."""
    s = unicodedata.normalize("NFKC", str(valor or ""))
    s = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in s)
    return " ".join(s.split())[:maximo]


def seguro_csv(valor):
    """Evita inyección de fórmulas al abrir el CSV en Excel/Sheets: antepone ' si el valor
    empieza (ignorando espacios Unicode iniciales) con = + - @, o con tab/CR/LF.
    Sin excepciones: un ' de más es inocuo; un '-' sin proteger puede ejecutarse."""
    if not isinstance(valor, str) or not valor:
        return valor
    i = 0   # salta espacios Unicode y caracteres invisibles (p. ej. BOM, ancho cero) iniciales
    while i < len(valor) and (valor[i].isspace() or unicodedata.category(valor[i]) == "Cf"):
        i += 1
    if valor[0] in "\t\r\n" or valor[i:i + 1] in ("=", "+", "-", "@"):
        return "'" + valor
    return valor


def df_seguro(df):
    """Copia del DataFrame con las columnas de texto neutralizadas (solo para CSV de análisis)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("str"):
            out[col] = out[col].map(seguro_csv)
    return out


def slug_codigo(texto):
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return (s or "categoria")[:30].strip("_")


def hash_categorias(cats):
    base = json.dumps([[c["codigo"], c["definicion"], c.get("criterios_inclusion", "")] for c in cats],
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


def cargar_datos():
    if not RESP_CSV.exists() or not CODEBOOK_CSV.exists():
        sys.exit("Faltan data/respuestas.csv y data/codebook.csv. Corre primero:\n"
                 "  python run_pipeline.py --preparar ARCHIVO_LIMPIO.csv CODEBOOK.csv")
    df = pd.read_csv(RESP_CSV, dtype=str, encoding="utf-8-sig")
    cb = pd.read_csv(CODEBOOK_CSV, dtype=str, encoding="utf-8-sig").set_index("variable")["pregunta"].to_dict()
    return df, cb


def tabla_respuestas(df, q):
    t = df[["ResponseId", q]].copy()
    t.columns = ["rid", "texto"]
    t["valida"] = t["texto"].map(es_respuesta)
    t["texto"] = t["texto"].fillna("").astype(str).str.strip()
    t["custom_id"] = q + "__" + t["rid"]
    bad = [c for c in t["custom_id"] if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", c)]
    if bad:
        raise ValueError(f"custom_id inválido (¿ResponseId raro?): {bad[:3]}")
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Preparación de datos (quita IP, correos, etc.: solo queda ResponseId + texto)
# ──────────────────────────────────────────────────────────────────────────────
def preparar(csv_limpio, codebook):
    df = pd.read_csv(csv_limpio, dtype=str, encoding="utf-8-sig")
    cb = pd.read_csv(codebook, dtype=str, encoding="utf-8-sig")
    faltan = [q for q in PREGUNTAS if q not in df.columns]
    if faltan:
        sys.exit(f"Columnas no encontradas en el CSV: {faltan}")
    DATA.mkdir(parents=True, exist_ok=True)
    sin_meta = [c for c in COLUMNAS_META if c not in df.columns]
    if sin_meta:
        sys.exit(f"Columnas de COLUMNAS_META no encontradas en el CSV: {sin_meta}")
    df[["ResponseId"] + PREGUNTAS].to_csv(RESP_CSV, index=False, encoding="utf-8-sig")
    # Aparte de las respuestas: estas columnas nunca llegan a los prompts, solo al archivo final.
    df[["ResponseId"] + COLUMNAS_META].to_csv(META_CSV, index=False, encoding="utf-8-sig")
    cb[cb["variable"].isin(PREGUNTAS)][["variable", "pregunta"]].to_csv(CODEBOOK_CSV, index=False, encoding="utf-8-sig")
    print(f"Listo: {RESP_CSV} ({len(df)} filas), {META_CSV} y {CODEBOOK_CSV}")


# ──────────────────────────────────────────────────────────────────────────────
# Prompts y esquemas
# ──────────────────────────────────────────────────────────────────────────────
def contexto():
    return leer_json(CONTEXTO_JSON, {}) or {}


def bloque_contexto(q, cb):
    """Contexto del cuestionario original (contexto_preguntas.json) para la pregunta q."""
    ctx = contexto()
    info = (ctx.get("preguntas") or {}).get(q, {})
    partes = [f"Encuesta: {ctx.get('encuesta') or CONTEXTO_ENCUESTA_BASE}"]
    if info.get("seccion"):
        partes.append(f"Sección del cuestionario: {info['seccion']}")
    if info.get("condicion"):
        partes.append(f"Quién la vio: {info['condicion']}")
    if info.get("situacion"):
        partes.append(f"Situación que se les presentó a los participantes (antes de la pregunta): {info['situacion']}")
    partes.append(f"Pregunta ({q}), tal como la leyeron los participantes: {cb.get(q, q)}")
    if info.get("notas"):
        partes.append(f"Notas para interpretar las respuestas: {info['notas']}")
    return "\n".join(partes)


def system_descubrir():
    regla = ("Una respuesta puede pertenecer a varias categorías a la vez, así que cada categoría debe poder "
             "aplicarse de forma independiente." if MULTI_ETIQUETA else
             "Cada respuesta recibirá una sola categoría, así que las categorías deben ser mutuamente excluyentes.")
    return f"""Eres un investigador cualitativo senior, experto en análisis temático inductivo de respuestas abiertas de encuestas en español.

Tu tarea: leer TODAS las respuestas a una pregunta abierta y construir el libro de códigos (categorías) que permita clasificar cada una.

Reglas:
- No hay número máximo ni mínimo de categorías: crea todas las necesarias para representar fielmente lo que dicen las personas. No fusiones ideas distintas solo para tener menos categorías, ni dividas una misma idea en categorías casi idénticas.
- Las categorías salen de los datos (enfoque inductivo), no de teorías previas ni de lo que "debería" responderse.
- Usa el contexto del cuestionario (descripción de la encuesta, situación presentada, quién vio la pregunta, notas) solo para interpretar bien las respuestas. Las categorías deben describir lo que las personas dicen o hacen, no repetir el enunciado ni la situación. Si el contexto menciona opciones cerradas de referencia, no las uses como categorías por defecto: crea una categoría solo si aparece en las respuestas.
- {regla}
- Cobertura: toda respuesta con contenido debe poder recibir al menos una categoría. Si aparecen en los datos, incluye categorías para respuestas vagas, de rechazo o fuera de tema (p. ej. "no usa IA", "no sabe").
- Cada categoría tiene: codigo (snake_case ASCII en minúsculas, único, máx. 30 caracteres), nombre (corto, en español), definicion (1-2 oraciones precisas), criterios_inclusion (cuándo SÍ aplicarla y cuándo NO, para distinguirla de categorías vecinas) y ejemplo (cita literal breve de una respuesta).
- Ordénalas aproximadamente de la más a la menos frecuente."""


def mensaje_descubrir(q, cb, textos):
    lista = "\n".join(f"[{i}] {t}" for i, t in enumerate(textos, 1))
    return f"{bloque_contexto(q, cb)}\n\nRespuestas ({len(textos)}):\n{lista}"


ESQUEMA_DESCUBRIR = {
    "type": "object",
    "properties": {"categorias": {"type": "array", "items": {
        "type": "object",
        "properties": {k: {"type": "string"} for k in
                       ("codigo", "nombre", "definicion", "criterios_inclusion", "ejemplo")},
        "required": ["codigo", "nombre", "definicion", "criterios_inclusion", "ejemplo"],
        "additionalProperties": False}}},
    "required": ["categorias"],
    "additionalProperties": False,
}


def libro_codigos(cats):
    # Los campos los generó un modelo a partir de texto de participantes (y el JSON es editable):
    # se tratan como datos no confiables → una línea por categoría y largo acotado.
    return "\n".join(f"- {linea(c['codigo'], 30)} ({linea(c['nombre'], 80)}): {linea(c['definicion'], 600)} "
                     f"Criterios: {linea(c.get('criterios_inclusion', ''), 800)}" for c in cats)


def system_clasificar(q, cb, cats):
    if MULTI_ETIQUETA:
        regla = ("Marca 1 en cada categoría cuyo criterio se cumple claramente en la respuesta y 0 en las demás. "
                 "Puede haber varias categorías en 1, o ninguna si nada encaja.")
    else:
        regla = "Elige la única categoría que mejor describe la respuesta, o \"ninguna\" si nada encaja."
    return f"""Eres un codificador cualitativo experto. Clasifica una respuesta abierta de encuesta usando EXCLUSIVAMENTE el siguiente libro de códigos.

{bloque_contexto(q, cb)}

Libro de códigos:
{libro_codigos(cats)}

Instrucciones:
- {regla}
- Basa tu decisión solo en lo que la respuesta dice, no en lo que podría implicar. El contexto del cuestionario sirve para entender a qué se refiere la respuesta (por ejemplo, la situación presentada), no para añadir categorías que la persona no expresó.
- Devuelve únicamente el JSON pedido."""


def esquema_clasificar(cats):
    codigos = [c["codigo"] for c in cats]
    if MULTI_ETIQUETA:
        return {"type": "object",
                "properties": {c: {"type": "integer", "enum": [0, 1]} for c in codigos},
                "required": codigos, "additionalProperties": False}
    return {"type": "object",
            "properties": {"categoria": {"type": "string", "enum": codigos + ["ninguna"]}},
            "required": ["categoria"], "additionalProperties": False}


def params_clasificar(q, cb, cats, texto):
    """Mismos parámetros para el lote y para la prueba síncrona."""
    return dict(
        model=MODELO_CLASIFICACION,
        max_tokens=MAX_TOKENS_CLASIFICACION,
        system=[{"type": "text", "text": system_clasificar(q, cb, cats),
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        messages=[{"role": "user", "content": f"Respuesta a clasificar:\n\n{texto}"}],
        output_config={"effort": ESFUERZO_CLASIFICACION,
                       "format": {"type": "json_schema", "schema": esquema_clasificar(cats)}},
    )


def validar_clasificacion(data, cats):
    """-> dict código→0/1 (lanza ValueError si la salida no es válida)."""
    codigos = [c["codigo"] for c in cats]
    if MULTI_ETIQUETA:
        if set(data) != set(codigos) or any(data[c] not in (0, 1) for c in codigos):
            raise ValueError(f"salida fuera del esquema: {str(data)[:120]}")
        return {c: int(data[c]) for c in codigos}
    elegido = data.get("categoria")
    if elegido not in codigos + ["ninguna"]:
        raise ValueError(f"categoría desconocida: {elegido}")
    return {c: int(c == elegido) for c in codigos}


# ──────────────────────────────────────────────────────────────────────────────
# Etapa 1: categorías con Opus
# ──────────────────────────────────────────────────────────────────────────────
class ErrorTerminal(RuntimeError):
    """Error que no cambia al reintentar (rechazo, max_tokens): no se reintenta."""


def texto_de(msg):
    if msg.stop_reason == "refusal":
        raise ErrorTerminal(f"El modelo rechazó la solicitud: {getattr(msg, 'stop_details', None)}")
    if msg.stop_reason == "max_tokens":
        raise ErrorTerminal("Se alcanzó max_tokens; sube MAX_TOKENS_* en la configuración.")
    return next(b.text for b in msg.content if b.type == "text")


def normalizar_categorias(brutas):
    vistos, cats = set(), []
    for c in brutas:
        cod = slug_codigo(c.get("codigo") or c.get("nombre"))
        base, n = cod, 2
        while cod in vistos or cod in RESERVADOS:
            cod, n = f"{base[:27]}_{n}", n + 1
        vistos.add(cod)
        cats.append({"codigo": cod, "nombre": linea(c.get("nombre"), 80), "definicion": linea(c.get("definicion"), 600),
                     "criterios_inclusion": linea(c.get("criterios_inclusion"), 800),
                     "ejemplo": linea(c.get("ejemplo"), 300)})
    if not cats:
        raise ValueError("El modelo no devolvió categorías.")
    return cats


def llamar_categorias(client, q, cb, textos):
    for intento in range(1, 4):
        try:
            with client.messages.stream(
                model=MODELO_CATEGORIAS,
                max_tokens=MAX_TOKENS_CATEGORIAS,
                thinking={"type": "adaptive"},
                system=system_descubrir(),
                messages=[{"role": "user", "content": mensaje_descubrir(q, cb, textos)}],
                output_config={"effort": ESFUERZO_CATEGORIAS,
                               "format": {"type": "json_schema", "schema": ESQUEMA_DESCUBRIR}},
            ) as stream:
                msg = stream.get_final_message()
            cats = normalizar_categorias(json.loads(texto_de(msg))["categorias"])
            return cats, msg.usage
        except (json.JSONDecodeError, KeyError, ValueError, StopIteration) as e:
            log.warning("%s: salida inválida de Opus (intento %d): %s", q, intento, e)
        except ErrorTerminal:
            raise
        except Exception as e:  # red/servidor: el SDK ya reintentó; esperamos y repetimos
            if intento == 3 or getattr(e, "status_code", 500) in (400, 401, 403, 404):
                raise
            log.warning("%s: error de API (intento %d): %s", q, intento, e)
            time.sleep(30 * intento)
    raise RuntimeError(f"{q}: no se pudo obtener un libro de códigos válido.")


def descubrir(client, df, cb, q):
    ruta = DIR_CAT / f"{q}.json"
    if ruta.exists():
        log.info("%s: categorías ya existen (%s) — se reutilizan", q, ruta.name)
        return leer_json(ruta)["categorias"]
    t = tabla_respuestas(df, q)
    textos = t.loc[t["valida"], "texto"].tolist()
    if len(textos) < 3:
        log.warning("%s: solo %d respuestas válidas; se omite", q, len(textos))
        return None
    log.info("%s: creando categorías con %s a partir de %d respuestas…", q, MODELO_CATEGORIAS, len(textos))
    cats, uso = llamar_categorias(client, q, cb, textos)
    guardar_json(ruta, {
        "pregunta": q, "texto_pregunta": cb.get(q), "n_respuestas": len(textos), "modelo": MODELO_CATEGORIAS,
        "esfuerzo": ESFUERZO_CATEGORIAS, "creado": ahora(), "hash": hash_categorias(cats),
        "uso": {"input": uso.input_tokens, "output": uso.output_tokens}, "categorias": cats})
    df_seguro(pd.DataFrame(cats)).to_csv(DIR_CAT / f"{q}.csv", index=False, encoding="utf-8-sig")
    log.info("%s: %d categorías creadas", q, len(cats))
    return cats


# ──────────────────────────────────────────────────────────────────────────────
# Etapa 2: clasificación con Sonnet en modo Batch
# ──────────────────────────────────────────────────────────────────────────────
def leer_raw(q):
    ruta = DIR_RAW / f"{q}.jsonl"
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def agregar_raw(q, filas):
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    with open(DIR_RAW / f"{q}.jsonl", "a", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def pendientes(t, q, h):
    raw = [r for r in leer_raw(q) if r.get("hash") == h]
    ok = {r["custom_id"] for r in raw if r["ok"]}
    fallos = Counter()
    for r in raw:
        if not r["ok"]:
            fallos[r["custom_id"]] += MAX_INTENTOS_POR_RESPUESTA if r.get("terminal") else 1
    return t[t["valida"] & ~t["custom_id"].isin(ok)
             & (t["custom_id"].map(lambda c: fallos[c]) < MAX_INTENTOS_POR_RESPUESTA)]


def construir_requests(df, cb, qs, cats_por_q):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reqs, meta = [], {}
    for q in qs:
        cats = cats_por_q.get(q)
        if not cats:
            continue
        h = hash_categorias(cats)
        t = tabla_respuestas(df, q)
        for fila in pendientes(t, q, h).itertuples():
            reqs.append(Request(custom_id=fila.custom_id,
                                params=MessageCreateParamsNonStreaming(**params_clasificar(q, cb, cats, fila.texto))))
            meta[fila.custom_id] = [q, fila.rid, h]
    return reqs, meta


def procesar_resultados(client, lote_id, meta):
    # Las categorías se leen del disco: el lote puede incluir preguntas que esta corrida no cubre.
    cats_por_q = {q: (leer_json(DIR_CAT / f"{q}.json") or {}).get("categorias") for q in {m[0] for m in meta.values()}}
    nuevos, resumen = {}, Counter()
    for r in client.messages.batches.results(lote_id):
        cid = r.custom_id
        if cid not in meta:
            continue
        q, rid, h = meta[cid]
        fila = {"custom_id": cid, "q": q, "rid": rid, "hash": h, "lote": lote_id, "ok": False}
        try:
            if r.result.type != "succeeded":
                detalle = getattr(getattr(r.result, "error", None), "error", None) or getattr(r.result, "error", None)
                if getattr(getattr(r.result, "error", None), "type", "") == "invalid_request":
                    fila["terminal"] = True            # no se reintenta: la solicitud es inválida
                raise RuntimeError(f"{r.result.type}: {detalle}")
            msg = r.result.message
            fila["data"] = validar_clasificacion(json.loads(texto_de(msg)), cats_por_q[q])
            fila["ok"] = True
        except Exception as e:
            fila["error"] = str(e)[:300]
        resumen["ok" if fila["ok"] else "error"] += 1
        nuevos.setdefault(q, []).append(fila)
    for q, filas in nuevos.items():
        agregar_raw(q, filas)
    return resumen


def esperar_lote(client, lote_id):
    while True:
        b = client.messages.batches.retrieve(lote_id)
        c = b.request_counts
        log.info("Lote %s: %s | procesando=%d ok=%d error=%d expirados=%d", lote_id, b.processing_status,
                 c.processing, c.succeeded, c.errored, c.expired)
        if b.processing_status == "ended":
            return
        time.sleep(SEGUNDOS_ENTRE_CONSULTAS)


def clasificar(client, df, cb, qs, cats_por_q):
    estado = leer_json(ESTADO, {})
    for ronda in range(1, MAX_RONDAS_LOTE + 1):
        lote = estado.get("lote")
        if lote:
            log.info("Retomando lote ya enviado %s (%d solicitudes)", lote["id"], len(lote["meta"]))
        else:
            reqs, meta = construir_requests(df, cb, qs, cats_por_q)
            if not reqs:
                log.info("Nada pendiente por clasificar.")
                return
            b = client.messages.batches.create(requests=reqs)
            lote = {"id": b.id, "enviado": ahora(), "meta": meta}
            estado["lote"] = lote
            guardar_json(ESTADO, estado)       # guardamos el id ANTES de esperar
            log.info("Ronda %d: lote %s enviado con %d solicitudes (%s)", ronda, b.id, len(reqs), MODELO_CLASIFICACION)
        esperar_lote(client, lote["id"])
        res = procesar_resultados(client, lote["id"], lote["meta"])
        log.info("Lote %s recogido: %s", lote["id"], dict(res))
        estado["lote"] = None
        estado.setdefault("lotes_completados", []).append({"id": lote["id"], "resultado": dict(res), "fin": ahora()})
        guardar_json(ESTADO, estado)
    log.warning("Se alcanzó MAX_RONDAS_LOTE; revisa 'estado_clasificacion' en la salida.")


# ──────────────────────────────────────────────────────────────────────────────
# Exportación
# ──────────────────────────────────────────────────────────────────────────────
def cargar_meta(df):
    """ResponseId + variables de identificación/demográficas (data/metadatos.csv), en el orden de df."""
    if not META_CSV.exists():
        log.warning("No existe %s: la salida solo tendrá ResponseId. Corre --preparar para incluir demográficos.", META_CSV)
        return df[["ResponseId"]].copy()
    m = pd.read_csv(META_CSV, dtype=str, encoding="utf-8-sig")
    if not m["ResponseId"].is_unique:
        raise ValueError("metadatos.csv tiene ResponseId duplicados")
    return df[["ResponseId"]].merge(m, on="ResponseId", how="left")


def exportar(df, cb, qs):
    DIR_CLA.mkdir(parents=True, exist_ok=True)
    meta = df_seguro(cargar_meta(df))      # ResponseId + demográficos; NO se usa al categorizar
    ancho = meta.copy()
    resumen = []
    for q in qs:
        info = leer_json(DIR_CAT / f"{q}.json")
        if not info:
            continue
        cats, h = info["categorias"], hash_categorias(info["categorias"])
        codigos = [c["codigo"] for c in cats]
        t = tabla_respuestas(df, q)
        ok = {r["custom_id"]: r["data"] for r in leer_raw(q) if r["ok"] and r.get("hash") == h}
        filas = []
        for fila in t.itertuples():
            estado = "sin_respuesta" if not fila.valida else ("ok" if fila.custom_id in ok else "error_o_pendiente")
            d = ok.get(fila.custom_id, {})
            filas.append({"ResponseId": fila.rid, "texto": fila.texto, "estado_clasificacion": estado,
                          **{c: d.get(c) for c in codigos}})
        out = df_seguro(pd.DataFrame(filas))     # texto de participantes: neutraliza fórmulas de Excel
        meta.merge(out, on="ResponseId", how="left").to_csv(DIR_CLA / f"{q}.csv", index=False, encoding="utf-8-sig")
        ancho = ancho.merge(out.rename(columns={c: f"{q}__{c}" for c in ["texto", "estado_clasificacion"] + codigos}),
                            on="ResponseId", how="left")
        n_ok = int((out["estado_clasificacion"] == "ok").sum())
        for c in cats:
            n = int(out[c["codigo"]].fillna(0).sum())
            resumen.append({"pregunta": q, "codigo": c["codigo"], "nombre": c["nombre"], "n": n,
                            "pct_de_clasificadas": round(100 * n / n_ok, 1) if n_ok else None})
        guardar_json(OUT / f"manifest_{q}.json", {
            "pregunta": q, "modelo_categorias": MODELO_CATEGORIAS, "esfuerzo_categorias": ESFUERZO_CATEGORIAS,
            "modelo_clasificacion": MODELO_CLASIFICACION, "esfuerzo_clasificacion": ESFUERZO_CLASIFICACION,
            "multi_etiqueta": MULTI_ETIQUETA, "hash_categorias": h, "exportado": ahora(),
            "conteo_estados": out["estado_clasificacion"].value_counts().to_dict()})
        log.info("%s: %s", q, out["estado_clasificacion"].value_counts().to_dict())
    ancho.to_csv(OUT / "clasificacion_todas_wide.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(resumen).to_csv(OUT / "resumen_categorias.csv", index=False, encoding="utf-8-sig")
    log.info("Exportado en %s", OUT)


# ──────────────────────────────────────────────────────────────────────────────
# Modos auxiliares
# ──────────────────────────────────────────────────────────────────────────────
def dry_run(df, cb, qs):
    cats_falsas = [{"codigo": "cat_a", "nombre": "A", "definicion": "def A", "criterios_inclusion": "x", "ejemplo": "e"},
                   {"codigo": "cat_b", "nombre": "B", "definicion": "def B", "criterios_inclusion": "y", "ejemplo": "e"}]
    total = 0
    print(f"Modelos: categorías={MODELO_CATEGORIAS} ({ESFUERZO_CATEGORIAS}) | clasificación={MODELO_CLASIFICACION} "
          f"(batch, {ESFUERZO_CLASIFICACION}) | multi-etiqueta={MULTI_ETIQUETA}\n")
    for q in qs:
        t = tabla_respuestas(df, q)
        n = int(t["valida"].sum())
        total += n
        aviso = "" if q in (contexto().get("preguntas") or {}) else "  ⚠ sin contexto en contexto_preguntas.json"
        chars = len(mensaje_descubrir(q, cb, t.loc[t["valida"], "texto"].tolist()))
        print(f"{q}: {n} válidas de {len(t)} | prompt Opus ≈ {chars:,} caracteres{aviso}")
        print(f"     {cb.get(q, '')[:110]}")
        p = params_clasificar(q, cb, cats_falsas, "texto de ejemplo")
        json.dumps(p, ensure_ascii=False)       # debe ser serializable
        assert json.dumps(esquema_clasificar(cats_falsas))
    reqs, meta = construir_requests(df, cb, qs, {q: cats_falsas for q in qs})
    print(f"\nSolicitudes de lote que se construirían (con categorías de ejemplo): {len(reqs)} (total válidas={total})")
    ids = [r["custom_id"] for r in reqs]
    assert len(ids) == len(set(ids)), "custom_id duplicados"
    print("OK: datos, prompts, esquemas y solicitudes de lote construidos sin llamar a la API.")


def prueba(client, df, cb, qs):
    q = next((x for x in qs if tabla_respuestas(df, x)["valida"].sum() >= 8), qs[0])
    t = tabla_respuestas(df, q)
    t = t[t["valida"]].head(12)
    print(f"PRUEBA con {q} ({len(t)} respuestas). No se guarda nada en resultados/.")
    cats, uso = llamar_categorias(client, q, cb, t["texto"].tolist())
    print(f"  Opus creó {len(cats)} categorías: {[c['codigo'] for c in cats]}")
    for fila in t.head(3).itertuples():
        msg = client.messages.create(**params_clasificar(q, cb, cats, fila.texto))
        d = validar_clasificacion(json.loads(texto_de(msg)), cats)
        print(f"  Sonnet → {fila.rid}: {[c for c, v in d.items() if v]}")
    print("PRUEBA OK: los parámetros de las dos etapas son aceptados por la API.")


def mostrar_estado(df, cb, qs):
    est = leer_json(ESTADO, {})
    print("Lote activo:", (est.get("lote") or {}).get("id", "ninguno"))
    for q in qs:
        t = tabla_respuestas(df, q)
        info = leer_json(DIR_CAT / f"{q}.json")
        if not info:
            print(f"{q}: sin categorías ({int(t['valida'].sum())} respuestas válidas)")
            continue
        h = hash_categorias(info["categorias"])
        ok = len({r["custom_id"] for r in leer_raw(q) if r["ok"] and r.get("hash") == h})
        print(f"{q}: {len(info['categorias'])} categorías | clasificadas {ok}/{int(t['valida'].sum())}")


def rehacer(qs):
    for q in qs:
        for p in (DIR_CAT / f"{q}.json", DIR_CAT / f"{q}.csv", DIR_RAW / f"{q}.jsonl"):
            p.unlink(missing_ok=True)
        log.info("%s: categorías y resultados borrados; se recrearán", q)


# ──────────────────────────────────────────────────────────────────────────────
def crear_cliente():
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    clave = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE")
    if not clave:
        sys.exit(f"Falta la API key. Crea {BASE / '.env'} con la línea:\nANTHROPIC_API_KEY=tu_clave")
    import anthropic
    return anthropic.Anthropic(api_key=clave, max_retries=5)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preparar", nargs=2, metavar=("CSV_LIMPIO", "CODEBOOK"), help="crea data/ (solo ResponseId + textos)")
    ap.add_argument("--dry-run", action="store_true", help="valida todo sin llamar a la API")
    ap.add_argument("--prueba", action="store_true", help="1 llamada a Opus + 3 a Sonnet (síncronas, centavos)")
    ap.add_argument("--solo-categorias", action="store_true", help="crea categorías y se detiene (para revisarlas/editarlas)")
    ap.add_argument("--estado", action="store_true", help="muestra el avance")
    ap.add_argument("--preguntas", nargs="+", metavar="Q", help="limitar a estas preguntas")
    ap.add_argument("--rehacer", nargs="+", metavar="Q", help="borra categorías y resultados de estas preguntas")
    args = ap.parse_args()

    if args.preparar:
        return preparar(*args.preparar)

    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG, encoding="utf-8")])
    df, cb = cargar_datos()
    qs = args.preguntas or PREGUNTAS
    desconocidas = [q for q in qs if q not in df.columns]
    if desconocidas:
        sys.exit(f"Preguntas desconocidas: {desconocidas}")

    if args.dry_run:
        return dry_run(df, cb, qs)
    if args.estado:
        return mostrar_estado(df, cb, qs)
    if args.rehacer:
        rehacer(args.rehacer)

    client = crear_cliente()
    if args.prueba:
        return prueba(client, df, cb, qs)

    cats_por_q = {}
    for q in qs:
        try:
            cats_por_q[q] = descubrir(client, df, cb, q)
        except Exception as e:                 # una pregunta que falla no detiene a las demás
            log.error("%s: falló la creación de categorías: %s", q, e)
    if args.solo_categorias:
        log.info("Categorías listas en %s. Edítalas si quieres y corre de nuevo sin --solo-categorias.", DIR_CAT)
        return
    clasificar(client, df, cb, qs, cats_por_q)
    exportar(df, cb, qs)


if __name__ == "__main__":
    main()
