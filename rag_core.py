from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence

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

AMBIGUOUS_PATTERNS = [
    r"^\s*qué resultados hay\s*\??\s*$",
    r"^\s*como vamos\s*\??\s*$",
    r"^\s*cómo vamos\s*\??\s*$",
    r"^\s*qué pasó\s*\??\s*$",
    r"^\s*que paso\s*\??\s*$",
    r"^\s*está bien o mal\s*\??\s*$",
    r"^\s*esta bien o mal\s*\??\s*$",
]

LISTING_HINTS = ["qué documentos", "que documentos", "documentos disponibles", "listado de documentos"]
COMPARISON_HINTS = ["compar", "cambió", "cambio", "entre", "histó", "años", "campaña", "campanas"]
RECENT_HINTS = ["actual", "hoy", "reciente", "último", "ultimo", "latest"]
ADVICE_HINTS = ["conviene", "recomend", "debería", "deberia", "vender", "retener"]


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
    cache_status: str
    top_campaigns: List[str]
    top_types: List[str]
    top_topics: List[str]


@dataclass
class RAGStore:
    metadata_df: pd.DataFrame
    chunks: List[ChunkRecord]
    embeddings: np.ndarray
    diagnostics: IngestionDiagnostics


def read_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


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
        snippet = text[start:end].strip()
        if snippet:
            chunks.append(snippet)
        if end >= len(text):
            break
        start += step
    return chunks


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return vectors / norms


def _serialize_chunks(chunks: Sequence[ChunkRecord]) -> List[Dict[str, Any]]:
    return [asdict(c) for c in chunks]


def _deserialize_chunks(items: Sequence[Dict[str, Any]]) -> List[ChunkRecord]:
    return [ChunkRecord(**item) for item in items]


def _fingerprint_corpus(df: pd.DataFrame, documents_dir: Path) -> str:
    h = hashlib.sha256()
    h.update(df.to_csv(index=False).encode("utf-8"))
    for name in sorted(df["Nuevo_nombre"].tolist()):
        p = documents_dir / name
        h.update(name.encode("utf-8"))
        if p.exists():
            stat = p.stat()
            h.update(str(stat.st_size).encode("utf-8"))
            h.update(str(int(stat.st_mtime)).encode("utf-8"))
        else:
            h.update(b"missing")
    return h.hexdigest()


def _get_cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "san_zenon_embeddings.npz", cache_dir / "san_zenon_chunks.json"


def _save_cache(
    cache_dir: Path,
    fingerprint: str,
    embeddings: np.ndarray,
    chunks: Sequence[ChunkRecord],
    diagnostics_payload: Dict[str, Any],
) -> str:
    npz_path, json_path = _get_cache_paths(cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, embeddings=embeddings, fingerprint=fingerprint)
        payload = {
            "fingerprint": fingerprint,
            "chunks": _serialize_chunks(chunks),
            "diagnostics": diagnostics_payload,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return "rebuild_and_cached"
    except Exception:
        return "rebuild_no_cache_write"


def _load_cache(cache_dir: Path, fingerprint: str) -> tuple[np.ndarray, List[ChunkRecord], Dict[str, Any]] | None:
    npz_path, json_path = _get_cache_paths(cache_dir)
    if not npz_path.exists() or not json_path.exists():
        return None

    try:
        npz = np.load(npz_path, allow_pickle=False)
        cached_fp = str(npz["fingerprint"])
        if cached_fp != fingerprint:
            return None
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if raw.get("fingerprint") != fingerprint:
            return None
        embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
        chunks = _deserialize_chunks(raw.get("chunks", []))
        if embeddings.shape[0] != len(chunks):
            return None
        diagnostics_payload = raw.get("diagnostics", {})
        return embeddings, chunks, diagnostics_payload
    except Exception:
        return None


def embed_texts(client: OpenAI, texts: Sequence[str], model: str = "text-embedding-3-small") -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    vectors: List[np.ndarray] = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=model, input=list(texts[i : i + batch_size]))
        vectors.append(np.array([item.embedding for item in response.data], dtype=np.float32))
    return _normalize_rows(np.vstack(vectors))


def classify_query_intent(question: str) -> str:
    q = question.lower().strip()
    if any(re.search(pattern, q) for pattern in AMBIGUOUS_PATTERNS):
        return "ambiguous"
    if any(h in q for h in LISTING_HINTS):
        return "document_listing"
    if any(h in q for h in COMPARISON_HINTS):
        return "historical_comparison"
    if any(h in q for h in ADVICE_HINTS):
        return "recommendation"
    return "factual_lookup"


def needs_recency_bonus(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in RECENT_HINTS)


def build_rag_store(
    client: OpenAI,
    index_path: Path,
    documents_dir: Path,
    campanas: Sequence[str] | None = None,
    tipos: Sequence[str] | None = None,
    temas: Sequence[str] | None = None,
    cache_dir: Path = Path(".cache"),
) -> RAGStore:
    df = load_index(index_path)
    filtered_df = df.copy()
    if campanas:
        filtered_df = filtered_df[filtered_df["Campaña"].isin(campanas)]
    if tipos:
        filtered_df = filtered_df[filtered_df["Tipo"].isin(tipos)]
    if temas:
        filtered_df = filtered_df[filtered_df["Tema_principal"].isin(temas)]

    fingerprint = _fingerprint_corpus(filtered_df, documents_dir)
    cached = _load_cache(cache_dir, fingerprint)
    if cached:
        embeddings, chunks, cached_diag = cached
        diagnostics = IngestionDiagnostics(
            documents_in_index=int(cached_diag.get("documents_in_index", len(filtered_df))),
            pdfs_found=int(cached_diag.get("pdfs_found", 0)),
            pdfs_missing=list(cached_diag.get("pdfs_missing", [])),
            unreadable_pdfs=list(cached_diag.get("unreadable_pdfs", [])),
            chunks_created=int(cached_diag.get("chunks_created", len(chunks))),
            metadata_fields_present=list(cached_diag.get("metadata_fields_present", REQUIRED_INDEX_COLUMNS)),
            cache_status="loaded_from_cache",
            top_campaigns=list(cached_diag.get("top_campaigns", [])),
            top_types=list(cached_diag.get("top_types", [])),
            top_topics=list(cached_diag.get("top_topics", [])),
        )
        return RAGStore(metadata_df=filtered_df, chunks=chunks, embeddings=embeddings, diagnostics=diagnostics)

    chunks: List[ChunkRecord] = []
    pdfs_found = 0
    pdfs_missing: List[str] = []
    unreadable_pdfs: List[str] = []

    for _, row in filtered_df.iterrows():
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

        for chunk_id, piece in enumerate(chunk_text(text, 1200, 180)):
            chunks.append(
                ChunkRecord(
                    text=piece,
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

    diagnostics_payload = {
        "documents_in_index": len(filtered_df),
        "pdfs_found": pdfs_found,
        "pdfs_missing": sorted(set(pdfs_missing)),
        "unreadable_pdfs": sorted(set(unreadable_pdfs)),
        "chunks_created": len(chunks),
        "metadata_fields_present": [c for c in REQUIRED_INDEX_COLUMNS if c in filtered_df.columns],
        "top_campaigns": filtered_df["Campaña"].value_counts().head(5).index.tolist(),
        "top_types": filtered_df["Tipo"].value_counts().head(5).index.tolist(),
        "top_topics": filtered_df["Tema_principal"].value_counts().head(5).index.tolist(),
    }
    cache_status = _save_cache(cache_dir, fingerprint, embeddings, chunks, diagnostics_payload)

    diagnostics = IngestionDiagnostics(
        documents_in_index=len(filtered_df),
        pdfs_found=pdfs_found,
        pdfs_missing=sorted(set(pdfs_missing)),
        unreadable_pdfs=sorted(set(unreadable_pdfs)),
        chunks_created=len(chunks),
        metadata_fields_present=[c for c in REQUIRED_INDEX_COLUMNS if c in filtered_df.columns],
        cache_status=cache_status,
        top_campaigns=filtered_df["Campaña"].value_counts().head(5).index.tolist(),
        top_types=filtered_df["Tipo"].value_counts().head(5).index.tolist(),
        top_topics=filtered_df["Tema_principal"].value_counts().head(5).index.tolist(),
    )

    return RAGStore(metadata_df=filtered_df, chunks=chunks, embeddings=embeddings, diagnostics=diagnostics)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\wáéíóúñ]+", text.lower())


def _keyword_overlap(query: str, chunk: ChunkRecord) -> float:
    q_terms = set(_tokenize(query))
    c_terms = set(_tokenize(" ".join([chunk.text, chunk.tema_principal, chunk.tipo, chunk.campana])))
    if not q_terms:
        return 0.0
    return len(q_terms & c_terms) / len(q_terms)


def _metadata_bonus(chunk: ChunkRecord, campanas: Sequence[str], tipos: Sequence[str], temas: Sequence[str]) -> float:
    bonus = 0.0
    if campanas and chunk.campana in campanas:
        bonus += 0.05
    if tipos and chunk.tipo in tipos:
        bonus += 0.05
    if temas and chunk.tema_principal in temas:
        bonus += 0.05
    return bonus


def _recency_bonus(chunk: ChunkRecord, enabled: bool) -> float:
    if not enabled:
        return 0.0
    m = re.search(r"(\d{4})", chunk.fecha)
    if not m:
        return 0.0
    year = int(m.group(1))
    return min(max((year - 2019) / 100, 0), 0.08)


def retrieve_top_k(
    store: RAGStore,
    query: str,
    client: OpenAI,
    k: int = 6,
    campanas: Sequence[str] | None = None,
    tipos: Sequence[str] | None = None,
    temas: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    if store.embeddings.shape[0] == 0:
        return []

    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    q = np.array(response.data[0].embedding, dtype=np.float32)
    q = q / max(np.linalg.norm(q), 1e-12)

    vector_scores = store.embeddings @ q
    recency_enabled = needs_recency_bonus(query)
    campanas = campanas or []
    tipos = tipos or []
    temas = temas or []

    ranked: List[Dict[str, Any]] = []
    for idx, vector_score in enumerate(vector_scores):
        ch = store.chunks[idx]
        keyword_score = _keyword_overlap(query, ch)
        meta_bonus = _metadata_bonus(ch, campanas, tipos, temas)
        recency = _recency_bonus(ch, recency_enabled)
        hybrid = float(vector_score) + 0.18 * keyword_score + meta_bonus + recency
        ranked.append(
            {
                "hybrid_score": hybrid,
                "vector_score": float(vector_score),
                "keyword_score": float(keyword_score),
                "metadata_bonus": float(meta_bonus),
                "recency_bonus": float(recency),
                "chunk": ch,
            }
        )

    ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return ranked[:k]


def answer_from_index_for_listing(store: RAGStore, question: str) -> Dict[str, Any]:
    q = question.lower()
    filtered = store.metadata_df

    if "sequía" in q or "sequia" in q:
        filtered = filtered[filtered["Tema_principal"].str.contains("SEQUIA|SEQUÍA", case=False, na=False)]
    if "tacto" in q or "preñez" in q or "prenez" in q:
        filtered = filtered[
            filtered["Tema_principal"].str.contains("TACTO|PRENEZ|PREÑEZ", case=False, na=False)
            | filtered["Nuevo_nombre"].str.contains("TACTO|PRENEZ|PREÑEZ", case=False, na=False)
        ]
    if "financier" in q:
        filtered = filtered[
            filtered["Tema_principal"].str.contains("FINAN", case=False, na=False)
            | filtered["Tipo"].str.contains("REU", case=False, na=False)
        ]

    rows = filtered[["Codigo", "Fecha", "Campaña", "Tipo", "Tema_principal", "Nuevo_nombre"]].head(25)
    if rows.empty:
        text = "No hay evidencia suficiente en el corpus documental disponible."
    else:
        bullets = [
            f"- {r.Codigo} | {r.Fecha} | {r.Campaña} | {r.Tipo} | {r.Tema_principal} | {r.Nuevo_nombre}"
            for r in rows.itertuples(index=False)
        ]
        text = "Documentos disponibles según el índice:\n" + "\n".join(bullets)

    return {"answer": text, "evidence": []}


def build_context(retrieved: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for item in retrieved:
        ch: ChunkRecord = item["chunk"]
        blocks.append(
            "\n".join(
                [
                    f"hybrid_score: {item['hybrid_score']:.4f}",
                    f"vector_score: {item['vector_score']:.4f}",
                    f"keyword_score: {item['keyword_score']:.4f}",
                    f"Codigo: {ch.codigo}",
                    f"Fecha: {ch.fecha}",
                    f"Campaña: {ch.campana}",
                    f"Tipo: {ch.tipo}",
                    f"Tema_principal: {ch.tema_principal}",
                    f"Nuevo_nombre: {ch.nuevo_nombre}",
                    f"Texto: {ch.text}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(
    client: OpenAI,
    system_prompt: str,
    question: str,
    retrieved: Sequence[Dict[str, Any]],
    intent: str,
) -> str:
    if intent == "ambiguous":
        return "¿Te referís a resultados ganaderos, agrícolas, financieros, reproductivos o a una campaña específica?"

    if not retrieved:
        return "No hay evidencia suficiente en el corpus documental disponible."

    context = build_context(retrieved)
    is_historical = intent == "historical_comparison"
    user_prompt = (
        "Respondé en español y solo con evidencia del contexto. "
        "Estructura obligatoria: Respuesta breve; Evidencia documental; Interpretación / lectura; "
        "Límites de la evidencia; Fuentes usadas. "
        "No sobre-cites: usar entre 3 y 5 fuentes salvo comparación histórica. "
        f"Comparación histórica solicitada: {'sí' if is_historical else 'no'}. "
        "Si la evidencia es débil o insuficiente, responder exactamente: "
        "No hay evidencia suficiente en el corpus documental disponible.\n\n"
        f"Pregunta:\n{question}\n\n"
        f"Contexto:\n{context}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content or ""
    return text.strip() or "No hay evidencia suficiente en el corpus documental disponible."


def format_sources(retrieved: Sequence[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    lines: List[str] = []
    for item in retrieved:
        ch: ChunkRecord = item["chunk"]
        if ch.nuevo_nombre in seen:
            continue
        seen.add(ch.nuevo_nombre)
        lines.append(
            f"- Codigo: {ch.codigo} | Fecha: {ch.fecha} | Campaña: {ch.campana} | Tipo: {ch.tipo} | Tema_principal: {ch.tema_principal} | Nuevo_nombre: {ch.nuevo_nombre}"
        )
    return lines
