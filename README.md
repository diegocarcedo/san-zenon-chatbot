# San Zenón — Chatbot documental

App RAG en Streamlit para consulta de documentos productivos y de gestión, optimizada para Streamlit Community Cloud.

## Qué mejoró

- Flujo tipo chat (panel principal solo mensajes + input fijo).
- Evidencia por respuesta en expander propio (score + metadatos + snippet).
- Detección de consultas ambiguas con pregunta de aclaración.
- Intención de consulta (listado documental, factual, comparación histórica, ambigua, recomendación).
- Scoring híbrido de retrieval (vector + keywords + metadata + recency opcional).
- Cache local de embeddings en `.cache/` con fingerprint de corpus.
- Diagnóstico ampliado (ingesta + cache + top metadatos).

## Estructura

- `app.py`
- `rag_core.py`
- `system_prompt.md`
- `data/00_INDEX/document_index.csv`
- `data/01_DOCUMENTS/`
- `qa/validate_app_sanity.py`
- `qa/run_ingestion_checks.py`
- `qa/evaluate_retrieval.py`
- `qa/sample_questions.md`
- `.github/workflows/sanity.yml`

## Despliegue en Streamlit Cloud

1. Conectar repo en Streamlit Community Cloud.
2. Confirmar `requirements.txt` y `runtime.txt` (`python-3.12`).
3. Definir secreto en Streamlit:

```toml
OPENAI_API_KEY = "tu_api_key"
```

4. Deploy.

## Dónde colocar PDFs

- Ubicación: `data/01_DOCUMENTS/`
- El nombre debe coincidir con `Nuevo_nombre` del índice.

## QA

```bash
python -m py_compile app.py rag_core.py qa/validate_app_sanity.py qa/run_ingestion_checks.py
python qa/validate_app_sanity.py
python qa/run_ingestion_checks.py --allow-missing-pdfs
```

Opcional (eval retrieval):

```bash
python qa/evaluate_retrieval.py
```

## Limitación del corpus

Este asistente responde solo sobre el corpus cargado. Para decisiones operativas completas, el corpus actual suele ser insuficiente.

## Documentos recomendados para agregar

- livestock inventories
- tacto reports
- paddock maps
- rainfall
- yields by lot
- financial records
- invoices / sales
- maintenance logs
- veterinary reports
- pasture records
- meeting minutes
