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

if not INDEX_PATH.exists():
    st.error(f"No se encontró el índice: {INDEX_PATH}")
    st.stop()

index_df = pd.read_csv(INDEX_PATH)
api_key = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None

with st.sidebar:
    st.header("Panel de control")
    campanas = st.multiselect("Campaña", sorted(index_df["Campaña"].dropna().astype(str).unique().tolist()))
    tipos = st.multiselect("Tipo", sorted(index_df["Tipo"].dropna().astype(str).unique().tolist()))
    temas = st.multiselect("Tema_principal", sorted(index_df["Tema_principal"].dropna().astype(str).unique().tolist()))

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    show_corpus_summary = st.button("Show corpus summary", use_container_width=True)

    with st.expander("Documentos cargados", expanded=False):
        st.dataframe(
            index_df[["Codigo", "Fecha", "Campaña", "Tipo", "Tema_principal", "Nuevo_nombre"]],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Limitaciones del corpus", expanded=False):
        st.markdown(
            """
El corpus actual es útil para explorar reportes y notas históricas, pero es insuficiente para decisiones operativas completas.

Datos recomendados:
- livestock inventories by date/category
- tacto reports across all years
- paddock maps and surfaces
- rainfall records
- crop yields by lot/campaign
- financial records
- purchase/sale invoices
- maintenance logs
- veterinary reports
- pasture/verdeo records
- meeting minutes and decision logs
"""
        )

    st.warning(
        "Este chatbot responde solo con la documentación cargada. "
        "No reemplaza asesoramiento técnico, contable, veterinario ni comercial."
    )

if not api_key:
    st.info("Configurá OPENAI_API_KEY en Streamlit Secrets para habilitar respuestas con modelo.")
    store = None
else:
    with st.spinner("Preparando índice, chunks y embeddings..."):
        store = build_rag_store(
            client=client,
            index_path=INDEX_PATH,
            documents_dir=DOCS_DIR,
            campanas=campanas,
            tipos=tipos,
            temas=temas,
        )

if store is not None:
    with st.sidebar:
        d = store.diagnostics
        st.markdown("### Estado rápido")
        st.caption(f"Documentos indexados: {d.documents_in_index}")
        st.caption(f"Chunks: {d.chunks_created}")
        st.caption(f"Cache: {d.cache_status}")

        with st.expander("Diagnóstico", expanded=False):
            st.write(f"documents in index: {d.documents_in_index}")
            st.write(f"PDFs found: {d.pdfs_found}")
            st.write(f"PDFs missing: {len(d.pdfs_missing)}")
            st.write(f"unreadable files: {len(d.unreadable_pdfs)}")
            st.write(f"chunks created: {d.chunks_created}")
            st.write(f"cache status: {d.cache_status}")
            st.write(f"metadata fields present: {d.metadata_fields_present}")
            if d.pdfs_missing:
                st.write("Missing:", d.pdfs_missing)
            if d.unreadable_pdfs:
                st.write("Unreadable:", d.unreadable_pdfs)

if not st.session_state.chat_history:
    st.markdown(
        """
### Bienvenido
**Qué podés preguntarle**
- Consultas sobre reportes, resultados históricos y documentos disponibles.

**Ejemplos de preguntas**
- ¿Qué documentos hay disponibles?
- ¿Qué documentos hablan de sequía?
- ¿Qué documentos mencionan tacto o preñez?
- ¿Qué documentos tratan temas financieros?
- ¿Cómo impactó la sequía en los distintos años?
- ¿Cuál fue el porcentaje de preñez en 2026?
- ¿Podés resumir la estrategia general del campo?
- ¿Qué datos faltan para tomar decisiones operativas?

**Limitaciones del corpus**
- Responde solo con documentación cargada y puede no cubrir toda la operación actual.
"""
    )

if show_corpus_summary:
    st.info(
        f"Corpus visible actual: {len(index_df)} documentos en índice. "
        "Usá ‘Documentos cargados’ en la barra lateral para ver el detalle."
    )


def evidence_label(answer_text: str, retrieved: list[dict]) -> str:
    if "No hay evidencia suficiente en el corpus documental disponible." in answer_text:
        return "Sin evidencia suficiente"
    if not retrieved:
        return "Evidencia parcial"
    strong_hits = [r for r in retrieved if r.get("hybrid_score", 0) >= 0.55]
    return "Evidencia fuerte" if len(strong_hits) >= 2 else "Evidencia parcial"


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.caption(f"Confianza: **{msg.get('evidence_label', 'Evidencia parcial')}**")
            if msg.get("evidence"):
                with st.expander("Evidencia usada", expanded=False):
                    for item in msg["evidence"]:
                        ch = item["chunk"]
                        st.markdown(
                            f"**score={item['hybrid_score']:.4f}** "
                            f"(vector={item['vector_score']:.4f}, keyword={item['keyword_score']:.4f}, "
                            f"metadata={item['metadata_bonus']:.4f}, recency={item['recency_bonus']:.4f})"
                        )
                        st.markdown(
                            f"`Codigo={ch.codigo}` | `Fecha={ch.fecha}` | `Campaña={ch.campana}` | `Tipo={ch.tipo}` | "
                            f"`Tema_principal={ch.tema_principal}` | `Nuevo_nombre={ch.nuevo_nombre}`"
                        )
                        st.write(ch.text[:850] + ("..." if len(ch.text) > 850 else ""))
                        st.divider()

question = st.chat_input("Consultá el corpus documental...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    intent = classify_query_intent(question)
    answer = "No hay evidencia suficiente en el corpus documental disponible."
    retrieved: list[dict] = []

    if store is None:
        answer = "No hay evidencia suficiente en el corpus documental disponible."
    elif intent == "document_listing":
        result = answer_from_index_for_listing(store, question)
        answer = result["answer"]
    elif intent == "ambiguous":
        answer = "¿Te referís a resultados ganaderos, agrícolas, financieros, reproductivos o a una campaña específica?"
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
            answer += "\n\n**Fuentes usadas**\n" + "\n".join(source_lines)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "evidence": retrieved,
            "evidence_label": evidence_label(answer, retrieved),
        }
    )
    st.rerun()
