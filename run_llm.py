"""Start the local model server and launch chat using project-local model storage."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def server_models():
    with urlopen("http://localhost:11434/api/tags", timeout=2) as response:
        return {model["name"] for model in json.load(response).get("models", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--age", type=int, default=35)
    parser.add_argument("--budget", choices=["low", "mid", "high", "unknown"], default="mid")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--voice", action="store_true", help="Launch the trained core voice client instead of text chat")
    parser.add_argument("--generator", choices=["ollama", "hybrid"], help="Voice defaults to hybrid; text defaults to ollama")
    args = parser.parse_args()
    local_exe = ROOT / ".runtime" / "ollama" / "ollama.exe"
    executable = str(local_exe) if local_exe.is_file() else shutil.which("ollama")
    if not executable:
        parser.error("Ollama is not installed. See docs/ollama_setup.md.")
    runtime = ROOT / ".runtime"
    models = runtime / "models"
    models.mkdir(parents=True, exist_ok=True)
    os.environ["OLLAMA_MODELS"] = str(models)
    os.environ["INSURANCE_LLM_MODEL"] = args.model
    os.environ["INSURANCE_LLM_URL"] = "http://localhost:11434/api/chat"
    os.environ.setdefault("INSURANCE_LLM_TIMEOUT", "120")
    try:
        available = server_models()
    except OSError:
        log_path = runtime / "ollama-server.log"
        temporary = runtime / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        server_env = os.environ.copy()
        server_env.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen([executable, "serve"], stdout=log, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                             env=server_env)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                available = server_models()
                break
            except OSError:
                time.sleep(1)
        else:
            parser.error(f"Ollama did not start. See {log_path}.")
    if args.model not in available:
        parser.error(f"Model is missing. In PowerShell run: & '{executable}' pull '{args.model}'")
    provider = args.generator or ("hybrid" if args.voice else "ollama")
    command = [sys.executable, str(ROOT / ("voice.py" if args.voice else "chat.py")), "--generator", provider,
               "--age", str(args.age), "--budget", args.budget]
    if args.debug:
        command.append("--debug")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
