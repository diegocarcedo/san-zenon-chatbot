from __future__ import annotations

from pathlib import Path

import streamlit as st
from openai import OpenAI

from rag_core import (
    answer_question,
    build_rag_store,
    format_sources,
    read_system_prompt,
    retrieve_top_k,
)

st.set_page_config(page_title="San Zenón — Chatbot documental", layout="wide")
st.title("San Zenón — Chatbot documental")

INDEX_PATH = Path("data/00_INDEX/document_index.csv")
DOCS_DIR = Path("data/01_DOCUMENTS")
SYSTEM_PROMPT_PATH = Path("system_prompt.md")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.warning("Configura OPENAI_API_KEY en Streamlit Secrets para habilitar el chatbot.")

client = OpenAI(api_key=api_key) if api_key else None

if not INDEX_PATH.exists():
    st.error(f"No se encontró el índice en: {INDEX_PATH}")
    st.stop()

import pandas as pd

index_df = pd.read_csv(INDEX_PATH)

with st.sidebar:
    st.header("Filtros")
    campanas = st.multiselect(
        "Campaña",
        options=sorted(index_df["Campaña"].dropna().astype(str).unique().tolist()),
    )
    tipos = st.multiselect(
        "Tipo",
        options=sorted(index_df["Tipo"].dropna().astype(str).unique().tolist()),
    )
    temas = st.multiselect(
        "Tema_principal",
        options=sorted(index_df["Tema_principal"].dropna().astype(str).unique().tolist()),
    )

chat_tab, diag_tab = st.tabs(["Chat", "Diagnóstico"])

store = None
if client:
    with st.spinner("Cargando corpus y generando embeddings..."):
        store = build_rag_store(
            client=client,
            index_path=INDEX_PATH,
            documents_dir=DOCS_DIR,
            campanas=campanas,
            tipos=tipos,
            temas=temas,
        )

with chat_tab:
    if not client:
        st.info("El chat está deshabilitado hasta definir OPENAI_API_KEY en Secrets.")
    elif store and store.diagnostics.pdfs_found == 0:
        st.info("No hay PDFs disponibles todavía en data/01_DOCUMENTS. El chatbot está listo cuando cargues documentos.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Escribe tu pregunta sobre el corpus documental...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not client or store is None:
                answer = "No hay evidencia suficiente en el corpus documental disponible."
                st.markdown(answer)
            else:
                retrieved = retrieve_top_k(store, question, client=client, k=6)
                st.session_state.last_retrieved = retrieved

                system_prompt = read_system_prompt(SYSTEM_PROMPT_PATH)
                answer = answer_question(
                    client=client,
                    system_prompt=system_prompt,
                    question=question,
                    retrieved=retrieved,
                )
                st.markdown(answer)

                sources = format_sources(retrieved)
                if sources:
                    st.markdown("**Fuentes citadas**")
                    for line in sources:
                        st.markdown(line)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.expander("Ver snippets recuperados"):
        retrieved = st.session_state.get("last_retrieved", [])
        if not retrieved:
            st.write("Aún no hay snippets recuperados.")
        else:
            for score, chunk in retrieved:
                st.markdown(
                    f"**{chunk.codigo}** | {chunk.fecha} | {chunk.campana} | {chunk.nuevo_nombre} | score={score:.4f}"
                )
                st.write(chunk.text[:800] + ("..." if len(chunk.text) > 800 else ""))
                st.divider()

with diag_tab:
    if store is None:
        st.info("Diagnóstico disponible al configurar OPENAI_API_KEY.")
    else:
        d = store.diagnostics
        st.metric("documents in index", d.documents_in_index)
        st.metric("PDFs found", d.pdfs_found)
        st.metric("PDFs missing", len(d.pdfs_missing))
        st.metric("unreadable PDFs", len(d.unreadable_pdfs))
        st.metric("chunks created", d.chunks_created)

        st.write("**metadata fields present**")
        st.write(d.metadata_fields_present)

        if d.pdfs_missing:
            st.write("**Listado de PDFs faltantes**")
            st.write(d.pdfs_missing)

        if d.unreadable_pdfs:
            st.write("**Listado de PDFs no legibles**")
            st.write(d.unreadable_pdfs)
