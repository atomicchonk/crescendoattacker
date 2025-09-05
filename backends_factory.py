# backends_factory.py
import os
from backends import LLMConfig, BaseLLMClient
from backends_openai import OpenAIClient
from backends_openai_compat import OpenAICompatClient
from backends_anthropic import AnthropicClient
from backends_gemini import GeminiClient
from backends_ollama import OllamaClient
from backends_hf_local import HFLocalClient

def make_client_from_env() -> tuple[BaseLLMClient, LLMConfig]:
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    model = os.getenv("MODEL", "gpt-4o-mini")
    temp = float(os.getenv("TEMP", "0.7"))
    max_toks = int(os.getenv("MAX_TOKENS", "400"))
    cfg = LLMConfig(provider=provider, model=model, temperature=temp, max_tokens=max_toks)

    if provider == "openai":
        return OpenAIClient(), cfg
    if provider in ("openai_compat", "openai-compatible", "compat"):
        return OpenAICompatClient(), cfg
    if provider == "anthropic":
        return AnthropicClient(), cfg
    if provider in ("google", "gemini"):
        return GeminiClient(), cfg
    if provider == "ollama":
        return OllamaClient(), cfg
    if provider in ("hf", "hf_local", "huggingface"):
        return HFLocalClient(), cfg

    raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")

