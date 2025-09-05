# backends_openai_compat.py
import os
from openai import OpenAI
from typing import List
from backends import BaseLLMClient, LLMConfig, Message

class OpenAICompatClient(BaseLLMClient):
    def __init__(self):
        base_url = os.getenv("OPENAI_COMPAT_BASE_URL")  # e.g. http://localhost:11434/v1 or another service
        api_key  = os.getenv("OPENAI_COMPAT_API_KEY", "not-needed")
        if not base_url:
            raise RuntimeError("OPENAI_COMPAT_BASE_URL is required for openai_compat provider")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: List[Message], cfg: LLMConfig) -> str:
        resp = self.client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            n=1,
        )
        return resp.choices[0].message.content

