from __future__ import annotations

from pathlib import Path
import re

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
        if (token == "Chroma" and "Chroma" in code) or (token != "Chroma" and token.lower() in lowered):
            print(f"ERROR: token prohibido detectado en app.py: {token}")
            return 1

    chat_inputs = len(re.findall(r"\bst\.chat_input\s*\(", code))
    if chat_inputs != 1:
        print(f"ERROR: app.py debe tener exactamente un st.chat_input y tiene {chat_inputs}")
        return 1

    if re.search(r"\bst\.tabs\s*\(", code) and ("Chat" in code and "Diagnóstico" in code):
        print("ERROR: se detectaron tabs Chat/Diagnóstico en app.py")
        return 1

    render_loops = len(re.findall(r"for\s+\w+\s+in\s+st\.session_state\.chat_history", code))
    if render_loops > 1:
        print("ERROR: posible render duplicado de chat history detectado")
        return 1

    if re.search(r"sk-[A-Za-z0-9]{10,}", code):
        print("ERROR: posible API key hardcodeada detectada")
        return 1

    print("OK: validate_app_sanity pasó correctamente")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
