# Cómo conectar la carpeta de Google Drive

Opción simple:
1. Descargar la carpeta `01_DOCUMENTS` como ZIP desde Google Drive.
2. Copiar los PDFs en `data/01_DOCUMENTS/`.
3. Copiar `document_index.csv` en `data/00_INDEX/`.
4. Ejecutar:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

Opción avanzada:
- Conectar Google Drive API.
- Leer la carpeta por `folder_id`.
- Descargar o sincronizar los PDFs automáticamente.
- Reindexar Chroma cada vez que haya nuevos documentos.

Folder ID actual:
`1UnGiepooKBZMtCmhGyUfiPx6RGUx_kiE`