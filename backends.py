# backends.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}

@dataclass
class LLMConfig:
    provider: str                  # "openai", "openai_compat", "anthropic", "gemini", "ollama", "hf_local"
    model: str
    temperature: float = 0.7
    max_tokens: int = 400
    timeout: Optional[int] = 30

class BaseLLMClient:
    def chat(self, messages: List[Message], cfg: LLMConfig) -> str:
        raise NotImplementedError

