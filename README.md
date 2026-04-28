# San Zenón — Chatbot documental (Streamlit + OpenAI + RAG)

Aplicación RAG ligera para consulta documental en español, diseñada para desplegarse fácilmente en **Streamlit Community Cloud** con dependencias mínimas.

## Características

- UI en Streamlit con dos pestañas: **Chat** y **Diagnóstico**.
- Ingesta desde:
  - Índice CSV: `data/00_INDEX/document_index.csv`
  - PDFs: `data/01_DOCUMENTS/`
- Columnas de metadatos preservadas:
  - `Codigo`, `Fecha`, `Tipo`, `Campaña`, `Tema_principal`, `Nombre_original`, `Nuevo_nombre`
- Extracción de texto PDF con `pypdf`.
- Chunking: **1200 caracteres** con **180 de solapamiento**.
- Embeddings con `text-embedding-3-small`.
- Vector store en memoria (numpy) con similitud coseno.
- Recuperación **top 6** fragmentos por consulta.
- Respuesta con `gpt-4o-mini`, temperatura `0.1`, en español.
- Citas de fuentes por: `Codigo`, `Fecha`, `Campaña`, `Nuevo_nombre`.
- Fallback explícito cuando falta evidencia:
  - “No hay evidencia suficiente en el corpus documental disponible.”

## Estructura

- `app.py`: interfaz Streamlit.
- `rag_core.py`: lógica de indexado, embeddings, retrieval y respuesta.
- `system_prompt.md`: prompt del sistema.
- `qa/validate_app_sanity.py`: chequeos anti-patrones en `app.py`.
- `qa/run_ingestion_checks.py`: validación de integridad de ingesta.
- `.github/workflows/sanity.yml`: pipeline básico de CI.

## Configuración local

1. Crear entorno virtual e instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Configurar API key (shell local):

```bash
export OPENAI_API_KEY="tu_api_key"
```

3. Ejecutar Streamlit:

```bash
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Subir este repositorio a GitHub.
2. En Streamlit Community Cloud, crear una app apuntando a `app.py`.
3. Verificar que se detecten:
   - `requirements.txt`
   - `runtime.txt` (`python-3.12`)
4. En **App settings → Secrets**, agregar:

```toml
OPENAI_API_KEY = "tu_api_key"
```

5. Reiniciar la app.

## Dónde colocar PDFs

- Carpeta: `data/01_DOCUMENTS/`
- El nombre de cada archivo debe coincidir con `Nuevo_nombre` en `data/00_INDEX/document_index.csv`.
- Si aún no hay PDFs, la app **no falla**: muestra mensaje claro y queda lista para ingestión.

## QA / validaciones

Ejecutar chequeos:

```bash
python -m py_compile app.py rag_core.py qa/validate_app_sanity.py qa/run_ingestion_checks.py
python qa/validate_app_sanity.py
python qa/run_ingestion_checks.py --allow-missing-pdfs
```

Qué valida cada script:

- `validate_app_sanity.py`
  - Falla si detecta en `app.py`:
    - `sys.version_info`
    - `langchain`
    - `chromadb`
    - `Chroma`
    - `load_vectorstore`
    - más de un `st.chat_input`

- `run_ingestion_checks.py`
  - Columnas requeridas del índice.
  - Existencia de carpeta de PDFs.
  - PDFs referenciados existentes.
  - PDFs legibles y con texto extraíble.
  - Conteo de chunks (1200/180).

## Limitaciones del corpus

Este chatbot solo puede responder usando el corpus cargado localmente. Si el contenido documental es parcial, incompleto o desactualizado, las respuestas tendrán el mismo límite.

Por diseño, ante evidencia insuficiente responde exactamente:

> No hay evidencia suficiente en el corpus documental disponible.

## Documentos recomendados para mejorar cobertura

Se recomienda incorporar, con consistencia de metadatos y nombre de archivo:

- inventarios ganaderos
- reportes de tacto
- mapas de potreros
- registros de lluvia
- rendimientos por lote
- registros financieros
- facturas
- logs de mantenimiento
- reportes veterinarios
- registros de pasturas
- actas/minutas de reuniones

## Dependencias permitidas

Este proyecto usa únicamente:

- streamlit
- openai
- pandas
- numpy
- pypdf
