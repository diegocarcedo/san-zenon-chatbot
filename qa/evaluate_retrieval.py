from __future__ import annotations

from pathlib import Path
import sys

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core import build_rag_store, classify_query_intent, retrieve_top_k

QUESTIONS = [
    "¿Qué documentos hablan de sequía?",
    "¿Qué documentos mencionan tacto o diagnóstico de preñez?",
    "¿Qué cambió entre el modelo productivo 2022 y 2026?",
    "¿Cómo impactó la sequía en los distintos años?",
]


def main() -> int:
    api_key = None
    try:
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        print("ERROR: define OPENAI_API_KEY para evaluar retrieval")
        return 1

    client = OpenAI(api_key=api_key)
    store = build_rag_store(
        client=client,
        index_path=Path("data/00_INDEX/document_index.csv"),
        documents_dir=Path("data/01_DOCUMENTS"),
    )

    for q in QUESTIONS:
        intent = classify_query_intent(q)
        top = retrieve_top_k(store=store, query=q, client=client, k=5)
        print("=" * 80)
        print(f"Q: {q}")
        print(f"Intent: {intent}")
        if not top:
            print("WARNING: sin resultados")
            continue
        for i, item in enumerate(top, start=1):
            ch = item["chunk"]
            print(
                f"{i}. {ch.nuevo_nombre} | {ch.fecha} | {ch.tema_principal} | "
                f"hybrid={item['hybrid_score']:.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
