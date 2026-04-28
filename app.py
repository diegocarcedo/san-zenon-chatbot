import os
import pandas as pd
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="San Zenón Chatbot", layout="wide")
st.title("San Zenón — Chatbot documental")

DATA_DIR = Path("data/01_DOCUMENTS")
INDEX_PATH = Path("data/00_INDEX/document_index.csv")
DB_DIR = Path("chroma_db")

SYSTEM_PROMPT = Path("system_prompt.md").read_text(encoding="utf-8")

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if DB_DIR.exists():
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    index = pd.read_csv(INDEX_PATH)
    docs = []
    for _, row in index.iterrows():
        pdf_path = DATA_DIR / row["Nuevo_nombre"]
        if not pdf_path.exists():
            continue

        reader = PdfReader(str(pdf_path))
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        metadata = row.to_dict()
        metadata["source_file"] = row["Nuevo_nombre"]
        docs.append(Document(page_content=text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )

vectorstore = load_vectorstore()

question = st.chat_input("Preguntá sobre San Zenón...")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    retrieved = retriever.invoke(question)

    context = "\n\n".join([
        f"FUENTE: {d.metadata.get('Codigo')} | {d.metadata.get('Fecha')} | {d.metadata.get('Tipo')} | {d.metadata.get('Tema_principal')}\n{d.page_content}"
        for d in retrieved
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Pregunta: {question}\n\nContexto documental recuperado:\n{context}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    answer = llm.invoke(prompt.format_messages(question=question, context=context)).content

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("Fuentes recuperadas"):
        for d in retrieved:
            st.write(d.metadata)

    st.session_state.history.append(("assistant", answer))