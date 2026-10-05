"""Download the configured Ollama models using its local API."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def pull(model):
    last_report = 0
    with requests.post(f"{BASE_URL}/api/pull", json={"model": model, "stream": True}, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            progress = json.loads(line)
            if "error" in progress:
                raise RuntimeError(f"{model}: {progress['error']}")
            status = progress.get("status", "")
            total = progress.get("total", 0)
            now = time.monotonic()
            if status == "success":
                print(f"{model}: downloaded and verified", flush=True)
                return
            if now - last_report >= 20:
                if total:
                    completed = progress.get("completed", 0)
                    print(f"{model}: {100 * completed / total:.1f}% ({completed / 1e6:.0f}/{total / 1e6:.0f} MB)", flush=True)
                else:
                    print(f"{model}: {status}", flush=True)
                last_report = now
    raise RuntimeError(f"{model}: download ended without success")


if __name__ == "__main__":
    models = {os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"), os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in executor.map(pull, sorted(models)):
            pass
