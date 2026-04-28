# San Zenón Chatbot — v0.1

Primer prototipo de chatbot documental para consultar informes históricos de San Zenón.

## Qué incluye
- Índice documental normalizado.
- Prompt base del asistente.
- Configuración RAG.
- App Streamlit lista para ejecutar localmente.
- Vector DB local con Chroma, creada al primer arranque.

## Estructura esperada

```text
data/
  00_INDEX/
    document_index.csv
  01_DOCUMENTS/
    SZ_INF_202007_GENERAL_FINAL_v01.pdf
    ...
system_prompt.md
app.py
requirements.txt
```

## Ejecutar

```bash
export OPENAI_API_KEY="tu_api_key"
pip install -r requirements.txt
streamlit run app.py
```

## Preguntas ejemplo
- ¿Cómo evolucionó la preñez entre campañas?
- ¿Qué mejoras de infraestructura se hicieron en 2025-2026?
- ¿Cuál es el modelo ganadero propuesto para San Zenón?
- ¿Qué superficie se destina a agricultura y qué rotación se propone?
- ¿Qué documentos hablan de sequía?