from __future__ import annotations

from pathlib import Path

import pandas as pd
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

INDEX_PATH = Path("data/00_INDEX/document_index.csv")
DOCS_DIR = Path("data/01_DOCUMENTS")
SYSTEM_PROMPT_PATH = Path("system_prompt.md")


def get_api_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


def evidence_label(retrieved):
    if not retrieved:
        return "Sin evidencia suficiente"
    top_score = retrieved[0][0]
    if len(retrieved) >= 3 and top_score >= 0.35:
        return "Evidencia fuerte"
    return "Evidencia parcial"


def render_evidence(retrieved):
    if not retrieved:
        st.write("No se recuperó evidencia documental suficiente.")
        return
    for i, (score, chunk) in enumerate(retrieved, start=1):
        st.markdown(
            f"**{i}. {chunk.codigo}** | {chunk.fecha} | Campaña {chunk.campana} | "
            f"{chunk.tipo} | score={score:.4f}"
        )
        st.caption(f"Archivo: {chunk.nuevo_nombre} | Tema: {chunk.tema_principal}")
        st.write(chunk.text[:900] + ("..." if len(chunk.text) > 900 else ""))
        st.divider()


def is_ambiguous(question: str) -> bool:
    q = question.lower().strip(" ¿?!.\n\t")
    ambiguous = {
        "qué resultados hay",
        "que resultados hay",
        "cómo vamos",
        "como vamos",
        "qué pasó",
        "que paso",
        "está bien o mal",
        "esta bien o mal",
    }
    return q in ambiguous


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("San Zenón — Chatbot documental")

if not INDEX_PATH.exists():
    st.error(f"No se encontró el índice en: {INDEX_PATH}")
    st.stop()

index_df = pd.read_csv(INDEX_PATH)
api_key = get_api_key()
client = OpenAI(api_key=api_key) if api_key else None

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

    st.divider()
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    with st.expander("Documentos cargados", expanded=False):
        cols = ["Codigo", "Fecha", "Campaña", "Tipo", "Tema_principal", "Nuevo_nombre"]
        st.dataframe(index_df[cols], use_container_width=True, hide_index=True)

    with st.expander("Limitaciones del corpus", expanded=False):
        st.warning(
            "Este chatbot responde solo con la documentación cargada. "
            "No reemplaza asesoramiento técnico, contable, veterinario ni comercial."
        )
        st.markdown(
            "El corpus actual sirve para explorar informes y notas históricas, "
            "pero no alcanza para decisiones operativas completas.\n\n"
            "Datos recomendados para una versión más robusta:\n"
            "- inventarios ganaderos por fecha y categoría;\n"
            "- tactos e informes veterinarios históricos;\n"
            "- mapas de potreros y superficies;\n"
            "- lluvias y clima;\n"
            "- rindes por lote y campaña;\n"
            "- registros financieros, ventas y compras;\n"
            "- mantenimiento e infraestructura;\n"
            "- actas/minutas de decisiones."
        )

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

with st.sidebar.expander("Diagnóstico", expanded=False):
    if store is None:
        st.info("Diagnóstico disponible cuando OPENAI_API_KEY esté configurada.")
    else:
        d = store.diagnostics
        st.metric("Documentos en índice", d.documents_in_index)
        st.metric("PDFs encontrados", d.pdfs_found)
        st.metric("PDFs faltantes", len(d.pdfs_missing))
        st.metric("Chunks creados", d.chunks_created)
        if d.pdfs_missing:
            st.write("PDFs faltantes")
            st.write(d.pdfs_missing)
        if d.unreadable_pdfs:
            st.write("PDFs no legibles")
            st.write(d.unreadable_pdfs)

if not api_key:
    st.info("Configura OPENAI_API_KEY en Streamlit Secrets para habilitar el chatbot.")
elif store and store.diagnostics.pdfs_found == 0:
    st.info("No hay PDFs disponibles todavía en data/01_DOCUMENTS. El chatbot está listo cuando cargues documentos.")

if not st.session_state.chat_history:
    st.markdown(
        "### Qué podés preguntarle\n"
        "- Qué documentos hay disponibles.\n"
        "- Qué documentos hablan de sequía, tacto, preñez, agricultura o finanzas.\n"
        "- Cómo evolucionó un tema entre campañas.\n"
        "- Qué datos faltan para tomar mejores decisiones.\n\n"
        "**Ejemplos:**\n"
        "- ¿Qué documentos hablan de sequía?\n"
        "- ¿Cuál fue el porcentaje de preñez en 2026?\n"
        "- ¿Podés resumir la estrategia general del campo?"
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            label = message.get("evidence_label")
            if label:
                st.caption(label)
            with st.expander("Evidencia usada", expanded=False):
                render_evidence(message.get("retrieved", []))

question = st.chat_input("Escribe tu pregunta sobre el corpus documental...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if is_ambiguous(question):
            answer = "¿Te referís a resultados ganaderos, agrícolas, financieros, reproductivos o a una campaña específica?"
            retrieved = []
        elif not client or store is None:
            answer = "No hay evidencia suficiente en el corpus documental disponible."
            retrieved = []
        else:
            retrieved = retrieve_top_k(store, question, client=client, k=6)
            system_prompt = read_system_prompt(SYSTEM_PROMPT_PATH)
            answer = answer_question(
                client=client,
                system_prompt=system_prompt,
                question=question,
                retrieved=retrieved,
            )

        st.markdown(answer)
        label = evidence_label(retrieved)
        st.caption(label)
        with st.expander("Evidencia usada", expanded=False):
            render_evidence(retrieved)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "retrieved": retrieved,
            "evidence_label": label,
        }
    )
