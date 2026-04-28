from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from pypdf import PdfReader

REQUIRED_INDEX_COLUMNS = [
    "Codigo",
    "Fecha",
    "Tipo",
    "Campaña",
    "Tema_principal",
    "Nombre_original",
    "Nuevo_nombre",
]


@dataclass
class ChunkRecord:
    text: str
    codigo: str
    fecha: str
    tipo: str
    campana: str
    tema_principal: str
    nombre_original: str
    nuevo_nombre: str
    chunk_id: int


@dataclass
class IngestionDiagnostics:
    documents_in_index: int
    pdfs_found: int
    pdfs_missing: List[str]
    unreadable_pdfs: List[str]
    chunks_created: int
    metadata_fields_present: List[str]


@dataclass
class RAGStore:
    metadata_df: pd.DataFrame
    chunks: List[ChunkRecord]
    embeddings: np.ndarray
    diagnostics: IngestionDiagnostics


def read_system_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def load_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    missing_cols = [c for c in REQUIRED_INDEX_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en index: {missing_cols}")

    for col in REQUIRED_INDEX_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> List[str]:
    if not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start += step
    return chunks


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return vectors / norms


def embed_texts(client: OpenAI, texts: Sequence[str], model: str = "text-embedding-3-small") -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    batch_size = 128
    all_vectors: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        response = client.embeddings.create(model=model, input=batch)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        all_vectors.append(vectors)

    matrix = np.vstack(all_vectors)
    return _normalize_rows(matrix)


def build_rag_store(
    client: OpenAI,
    index_path: Path,
    documents_dir: Path,
    campanas: Sequence[str] | None = None,
    tipos: Sequence[str] | None = None,
    temas: Sequence[str] | None = None,
) -> RAGStore:
    df = load_index(index_path)

    if campanas:
        df = df[df["Campaña"].isin(campanas)]
    if tipos:
        df = df[df["Tipo"].isin(tipos)]
    if temas:
        df = df[df["Tema_principal"].isin(temas)]

    chunks: List[ChunkRecord] = []
    pdfs_found = 0
    pdfs_missing: List[str] = []
    unreadable_pdfs: List[str] = []

    for _, row in df.iterrows():
        pdf_path = documents_dir / row["Nuevo_nombre"]
        if not pdf_path.exists():
            pdfs_missing.append(row["Nuevo_nombre"])
            continue

        pdfs_found += 1
        try:
            text = extract_pdf_text(pdf_path)
        except Exception:
            unreadable_pdfs.append(row["Nuevo_nombre"])
            continue

        local_chunks = chunk_text(text, chunk_size=1200, overlap=180)
        for chunk_id, chunk in enumerate(local_chunks):
            chunks.append(
                ChunkRecord(
                    text=chunk,
                    codigo=row["Codigo"],
                    fecha=row["Fecha"],
                    tipo=row["Tipo"],
                    campana=row["Campaña"],
                    tema_principal=row["Tema_principal"],
                    nombre_original=row["Nombre_original"],
                    nuevo_nombre=row["Nuevo_nombre"],
                    chunk_id=chunk_id,
                )
            )

    texts = [c.text for c in chunks]
    embeddings = embed_texts(client, texts) if texts else np.zeros((0, 1536), dtype=np.float32)

    diagnostics = IngestionDiagnostics(
        documents_in_index=len(df),
        pdfs_found=pdfs_found,
        pdfs_missing=sorted(set(pdfs_missing)),
        unreadable_pdfs=sorted(set(unreadable_pdfs)),
        chunks_created=len(chunks),
        metadata_fields_present=[c for c in REQUIRED_INDEX_COLUMNS if c in df.columns],
    )

    return RAGStore(metadata_df=df, chunks=chunks, embeddings=embeddings, diagnostics=diagnostics)


def retrieve_top_k(store: RAGStore, query: str, client: OpenAI, k: int = 6) -> List[Tuple[float, ChunkRecord]]:
    if store.embeddings.shape[0] == 0:
        return []

    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    q = np.array(response.data[0].embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    q = q / q_norm

    scores = store.embeddings @ q
    top_idx = np.argsort(scores)[-k:][::-1]
    return [(float(scores[i]), store.chunks[int(i)]) for i in top_idx]


def build_context(retrieved: Sequence[Tuple[float, ChunkRecord]]) -> str:
    parts: List[str] = []
    for score, chunk in retrieved:
        parts.append(
            "\n".join(
                [
                    f"[score={score:.4f}]",
                    f"Codigo: {chunk.codigo}",
                    f"Fecha: {chunk.fecha}",
                    f"Campaña: {chunk.campana}",
                    f"Nuevo_nombre: {chunk.nuevo_nombre}",
                    f"Texto: {chunk.text}",
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def answer_question(
    client: OpenAI,
    system_prompt: str,
    question: str,
    retrieved: Sequence[Tuple[float, ChunkRecord]],
) -> str:
    if not retrieved:
        return "No hay evidencia suficiente en el corpus documental disponible."

    context = build_context(retrieved)
    user_prompt = (
        "Responde en español usando únicamente la evidencia del contexto. "
        "Si la evidencia es insuficiente, responde exactamente: "
        "No hay evidencia suficiente en el corpus documental disponible.\n\n"
        f"Pregunta:\n{question}\n\n"
        f"Contexto documental:\n{context}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    output = response.choices[0].message.content or ""
    return output.strip() or "No hay evidencia suficiente en el corpus documental disponible."


def format_sources(retrieved: Sequence[Tuple[float, ChunkRecord]]) -> List[str]:
    unique: Dict[str, ChunkRecord] = {}
    for _, chunk in retrieved:
        key = chunk.nuevo_nombre
        if key not in unique:
            unique[key] = chunk

    lines: List[str] = []
    for chunk in unique.values():
        lines.append(
            f"- Codigo: {chunk.codigo} | Fecha: {chunk.fecha} | Campaña: {chunk.campana} | Nuevo_nombre: {chunk.nuevo_nombre}"
        )
    return lines
