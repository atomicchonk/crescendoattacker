# backends_anthropic.py
import os
from typing import List
try:
    import anthropic
except ImportError:
    anthropic = None

from backends import BaseLLMClient, LLMConfig, Message

class AnthropicClient(BaseLLMClient):
    def __init__(self):
        if anthropic is None:
            raise RuntimeError("anthropic package not installed. pip install anthropic")
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def chat(self, messages: List[Message], cfg: LLMConfig) -> str:
        # Convert OpenAI-style messages to Anthropic "messages" format
        # Keep only the latest system (if any) as system prompt
        system_prompt = ""
        converted = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            elif m["role"] in ("user", "assistant"):
                converted.append({"role": m["role"], "content": m["content"]})

        resp = self.client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=system_prompt or None,
            messages=converted,
        )
        # Anthropic returns a list of content blocks
        return "".join([blk.text for blk in resp.content if getattr(blk, "type", "") == "text"])

