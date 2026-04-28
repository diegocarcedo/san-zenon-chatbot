from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from pypdf import PdfReader

REQUIRED_COLUMNS = [
    "Codigo",
    "Fecha",
    "Tipo",
    "Campaña",
    "Tema_principal",
    "Nombre_original",
    "Nuevo_nombre",
]

INDEX_PATH = Path("data/00_INDEX/document_index.csv")
PDF_DIR = Path("data/01_DOCUMENTS")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> List[str]:
    if not text.strip():
        return []
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


def extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pieces = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            pieces.append(t)
    return "\n\n".join(pieces).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-pdfs", action="store_true")
    args = parser.parse_args()

    errors: List[str] = []

    if not INDEX_PATH.exists():
        errors.append(f"No existe índice: {INDEX_PATH}")
        print("\n".join(errors))
        return 1

    df = pd.read_csv(INDEX_PATH)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Faltan columnas requeridas: {missing_cols}")

    if not PDF_DIR.exists() or not PDF_DIR.is_dir():
        errors.append(f"No existe carpeta de PDFs: {PDF_DIR}")

    referenced = df["Nuevo_nombre"].dropna().astype(str).str.strip().tolist() if "Nuevo_nombre" in df.columns else []

    missing_pdfs = []
    unreadable = []
    no_text = []
    chunk_count = 0

    for name in referenced:
        p = PDF_DIR / name
        if not p.exists():
            missing_pdfs.append(name)
            continue
        try:
            text = extract_text(p)
        except Exception:
            unreadable.append(name)
            continue
        if not text:
            no_text.append(name)
        chunk_count += len(chunk_text(text))

    if missing_pdfs and not args.allow_missing_pdfs:
        errors.append(f"PDFs referenciados faltantes ({len(missing_pdfs)}): {missing_pdfs}")

    if unreadable:
        errors.append(f"PDFs no legibles ({len(unreadable)}): {unreadable}")

    if no_text:
        errors.append(f"PDFs sin texto extraíble ({len(no_text)}): {no_text}")

    print("Resumen de ingestión:")
    print(f"- Filas índice: {len(df)}")
    print(f"- PDFs referenciados: {len(referenced)}")
    print(f"- PDFs faltantes: {len(missing_pdfs)}")
    print(f"- PDFs no legibles: {len(unreadable)}")
    print(f"- PDFs sin texto: {len(no_text)}")
    print(f"- Chunks totales (1200/180): {chunk_count}")

    if errors:
        print("\nErrores:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("OK: run_ingestion_checks pasó correctamente")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
