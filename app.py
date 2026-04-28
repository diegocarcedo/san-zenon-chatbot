from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from rag_core import (
    answer_from_index_for_listing,
    answer_question,
    build_rag_store,
    classify_query_intent,
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

api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.warning("Configurá OPENAI_API_KEY en Streamlit Secrets para habilitar respuestas con modelo.")

client = OpenAI(api_key=api_key) if api_key else None

if not INDEX_PATH.exists():
    st.error(f"No se encontró el índice: {INDEX_PATH}")
    st.stop()

index_df = pd.read_csv(INDEX_PATH)

with st.sidebar:
    st.header("Filtros")
    campanas = st.multiselect("Campaña", sorted(index_df["Campaña"].dropna().astype(str).unique().tolist()))
    tipos = st.multiselect("Tipo", sorted(index_df["Tipo"].dropna().astype(str).unique().tolist()))
    temas = st.multiselect("Tema_principal", sorted(index_df["Tema_principal"].dropna().astype(str).unique().tolist()))
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

store = None
if client:
    with st.spinner("Preparando índice, chunks y embeddings..."):
        store = build_rag_store(
            client=client,
            index_path=INDEX_PATH,
            documents_dir=DOCS_DIR,
            campanas=campanas,
            tipos=tipos,
            temas=temas,
        )

chat_tab, diagnostics_tab = st.tabs(["Chat", "Diagnostics"])

with chat_tab:
    if not client:
        st.info("Chat disponible cuando configures OPENAI_API_KEY.")
    elif store and store.diagnostics.pdfs_found == 0:
        st.info("No hay PDFs en data/01_DOCUMENTS. La app no falla; solo espera documentos.")

    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("evidence"):
                with st.expander(f"Evidencia recuperada (respuesta {idx // 2 + 1})", expanded=False):
                    for item in msg["evidence"]:
                        ch = item["chunk"]
                        st.markdown(
                            (
                                f"**score híbrido={item['hybrid_score']:.4f} | vector={item['vector_score']:.4f} | "
                                f"keyword={item['keyword_score']:.4f} | metadata={item['metadata_bonus']:.4f} | "
                                f"recency={item['recency_bonus']:.4f}**"
                            )
                        )
                        st.markdown(
                            f"`Codigo={ch.codigo}` | `Fecha={ch.fecha}` | `Campaña={ch.campana}` | "
                            f"`Tipo={ch.tipo}` | `Tema_principal={ch.tema_principal}` | `Nuevo_nombre={ch.nuevo_nombre}`"
                        )
                        st.write(ch.text[:900] + ("..." if len(ch.text) > 900 else ""))
                        st.divider()

    question = st.chat_input("Consultá el corpus documental...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        intent = classify_query_intent(question)

        answer = "No hay evidencia suficiente en el corpus documental disponible."
        retrieved = []

        if store is None:
            answer = "No hay evidencia suficiente en el corpus documental disponible."
        elif intent == "document_listing":
            result = answer_from_index_for_listing(store, question)
            answer = result["answer"]
        elif intent == "ambiguous":
            answer = "¿Te referís a resultados ganaderos, agrícolas, financieros, reproductivos o de una campaña específica?"
        else:
            top_k = 8 if intent == "historical_comparison" else 5
            retrieved = retrieve_top_k(
                store=store,
                query=question,
                client=client,
                k=top_k,
                campanas=campanas,
                tipos=tipos,
                temas=temas,
            )
            answer = answer_question(
                client=client,
                system_prompt=read_system_prompt(SYSTEM_PROMPT_PATH),
                question=question,
                retrieved=retrieved,
                intent=intent,
            )

            source_lines = format_sources(retrieved)
            if source_lines:
                answer = answer + "\n\n**Fuentes usadas**\n" + "\n".join(source_lines)

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "evidence": retrieved})
        st.rerun()

with diagnostics_tab:
    if store is None:
        st.info("Diagnostics disponibles al configurar OPENAI_API_KEY.")
    else:
        d = store.diagnostics
        st.metric("documents in index", d.documents_in_index)
        st.metric("PDFs found", d.pdfs_found)
        st.metric("PDFs missing", len(d.pdfs_missing))
        st.metric("unreadable PDFs", len(d.unreadable_pdfs))
        st.metric("chunks created", d.chunks_created)
        st.metric("cache status", d.cache_status)

        st.write("**metadata fields present**")
        st.write(d.metadata_fields_present)
        st.write("**top campaigns**", d.top_campaigns)
        st.write("**top types**", d.top_types)
        st.write("**top topics**", d.top_topics)

        st.warning("El corpus actual es útil para exploración, pero insuficiente para decisiones operativas completas.")
        st.markdown(
            """
**Datos recomendados para incorporar**
- inventories by category/date
- tacto reports across all years
- rainfall data
- paddock maps
- crop yields by lot
- financial records
- invoices/sales
- maintenance logs
- veterinary reports
- meeting minutes
"""
        )

        if d.pdfs_missing:
            st.write("**missing PDFs list**")
            st.write(d.pdfs_missing)
        if d.unreadable_pdfs:
            st.write("**unreadable PDFs list**")
            st.write(d.unreadable_pdfs)
