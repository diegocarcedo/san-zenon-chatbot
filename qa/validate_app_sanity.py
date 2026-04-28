from __future__ import annotations

from pathlib import Path
import re
import sys

APP_PATH = Path("app.py")

FORBIDDEN_TOKENS = [
    "sys.version_info",
    "langchain",
    "chromadb",
    "Chroma",
    "load_vectorstore",
]


def main() -> int:
    if not APP_PATH.exists():
        print("ERROR: app.py no existe")
        return 1

    code = APP_PATH.read_text(encoding="utf-8")
    lowered = code.lower()

    for token in FORBIDDEN_TOKENS:
        token_found = token.lower() in lowered if token != "Chroma" else "Chroma" in code
        if token_found:
            print(f"ERROR: Se encontró token prohibido en app.py: {token}")
            return 1

    chat_inputs = len(re.findall(r"\bst\.chat_input\s*\(", code))
    if chat_inputs > 1:
        print(f"ERROR: app.py contiene más de un st.chat_input ({chat_inputs})")
        return 1

    print("OK: validate_app_sanity pasó correctamente")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
