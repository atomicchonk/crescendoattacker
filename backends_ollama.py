# backends_ollama.py
import os, requests, json
from typing import List
from backends import BaseLLMClient, LLMConfig, Message

class OllamaClient(BaseLLMClient):
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # We'll use the /api/chat endpoint (simple stream=False)
        self.url = f"{self.base_url}/api/chat"

    def chat(self, messages: List[Message], cfg: LLMConfig) -> str:
        payload = {
            "model": cfg.model,          # e.g., "llama3"
            "temperature": cfg.temperature,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": cfg.max_tokens}
        }
        r = requests.post(self.url, json=payload, timeout=cfg.timeout or 30)
        r.raise_for_status()
        data = r.json()
        # Ollama returns { "message": { "role": "assistant", "content": "..." }, ... }
        return data.get("message", {}).get("content", "")

