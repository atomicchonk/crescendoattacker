# backends_openai.py
import os
from openai import OpenAI
from typing import List
from backends import BaseLLMClient, LLMConfig, Message

class OpenAIClient(BaseLLMClient):
    def __init__(self):
        # OPENAI_API_KEY taken from env by the SDK automatically,
        # or you can pass api_key=os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()

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

