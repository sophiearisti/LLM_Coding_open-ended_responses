# Pipeline automático: categorías (Opus 5.5) → clasificación (Sonnet 5.5, Batch)

Para cada pregunta abierta:

1. **Opus 5.5** lee todas las respuestas y crea las categorías que haga falta (sin límite).
2. **Sonnet 5.5 en modo Batch** (50 % más barato) clasifica cada respuesta con esas categorías.
3. Todo queda en `resultados/`.

## 1. Dónde pegar la API key

Crea el archivo **`pipeline_categorias/.env`** (copia `.env.example`) con una sola línea:

```
ANTHROPIC_API_KEY=sk-ant-...
```

- Alternativa: la variable `CLAUDE=` también funciona (es la que usa `tool/.env`).
- `.env`, `data/` y `resultados/` están en `.gitignore`: no se suben a git. Ojo: la carpeta está en
  Dropbox, así que el `.env` sí se sincroniza con tu Dropbox.

## 2. Instalación (una vez)

```powershell
cd pipeline_categorias
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(Usa un venv propio para no tocar el de la app de Streamlit.)

## 3. Datos (una vez; ya está hecho para el archivo actual)

```powershell
python run_pipeline.py --preparar RUTA\Diagnostico_IA_docentes_clean.csv RUTA\Diagnostico_IA_docentes_codebook.csv
```

Crea `data/respuestas.csv` con **solo** `ResponseId` + las preguntas abiertas (sin IP, correos ni coordenadas).

## 4. Correr

```powershell
python run_pipeline.py --dry-run     # sin API: valida datos, prompts, esquemas y solicitudes
python run_pipeline.py --prueba      # 1 llamada a Opus + 3 a Sonnet: confirma que la API acepta los parámetros
python run_pipeline.py               # corrida completa (cuesta dinero)
```

Otras opciones:

| Opción | Para qué |
|---|---|
| `--solo-categorias` | Crea las categorías y se detiene. Puedes editar `resultados/categorias/Q14.json` y luego correr sin la bandera. |
| `--preguntas Q14 Q15` | Limita a algunas preguntas. |
| `--rehacer Q14` | Borra categorías y resultados de esa pregunta para recrearlos. |
| `--estado` | Muestra el avance. |

## 5. Dejarlo corriendo solo (o con un agente)

El script es **reanudable e idempotente**: guarda el id del lote antes de esperarlo y nunca reenvía lo ya hecho.
Si se corta (apagón, cierre de terminal), vuelve a ejecutar el mismo comando y continúa.
Un agente solo debe repetir `python run_pipeline.py` hasta que `--estado` muestre todas las preguntas
completas (`clasificadas N/N`). Los logs están en `resultados/pipeline.log`.

## Salidas (`resultados/`)

| Archivo | Contenido |
|---|---|
| `categorias/Q*.json` / `.csv` | Libro de códigos (código, nombre, definición, criterios, ejemplo) |
| `clasificacion/Q*.csv` | `ResponseId`, texto, estado y una columna 0/1 por categoría |
| `clasificacion_todas_wide.csv` | Todo unido por `ResponseId` (columnas `Q14__codigo`) |
| `resumen_categorias.csv` | Frecuencia de cada categoría por pregunta |
| `clasificacion_raw/Q*.jsonl` | Respuestas crudas del lote (los resultados del Batch expiran a los 29 días) |
| `manifest_Q*.json` | Modelos, esfuerzo, hash de categorías |

`estado_clasificacion`: `ok`, `sin_respuesta` (vacío o relleno como `-`), `error_o_pendiente` (falló tras reintentos).

## Supuestos y límites

- **Multi-etiqueta** por defecto (una respuesta puede tener varias categorías). Cámbialo con `MULTI_ETIQUETA` en `run_pipeline.py`.
- **Contexto del cuestionario:** `contexto_preguntas.json` (tomado del cuestionario original en Word) describe la encuesta
  y, por pregunta, la sección, quién la vio, las situaciones de Q18/Q21/Q24/Q27 y notas de interpretación. Se inyecta en
  los prompts de crear y de asignar categorías. Edítalo ahí si quieres afinar los prompts.
  Las opciones cerradas de cada situación solo se dan como referencia; el modelo no las usa como categorías por defecto.
- Si editas un libro de códigos después de clasificar, los resultados anteriores se descartan automáticamente (hash) y se reclasifica.
