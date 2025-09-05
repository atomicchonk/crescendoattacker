# backends_hf_local.py
from typing import List
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None

from backends import BaseLLMClient, LLMConfig, Message

class HFLocalClient(BaseLLMClient):
    def __init__(self):
        if AutoModelForCausalLM is None:
            raise RuntimeError("transformers not installed. pip install transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cpu")
        # Lazy init, load on first call to avoid long import at startup
        self.nlp = None
        self.loaded_model = None

    def _ensure_loaded(self, model_name: str):
        if self.loaded_model == model_name and self.nlp is not None:
            return
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        self.nlp = pipeline("text-generation", model=mdl, tokenizer=tok)
        self.loaded_model = model_name

    def chat(self, messages: List[Message], cfg: LLMConfig) -> str:
        self._ensure_loaded(cfg.model)
        # Simple concatenation; for chat-tuned models you may need special formatting
        prompt = ""
        for m in messages:
            if m["role"] == "system":
                prompt += f"[System] {m['content']}\n"
            elif m["role"] == "user":
                prompt += f"[User] {m['content']}\n"
            elif m["role"] == "assistant":
                prompt += f"[Assistant] {m['content']}\n"
        prompt += "[Assistant] "

        out = self.nlp(prompt, max_new_tokens=cfg.max_tokens, temperature=cfg.temperature, do_sample=True)[0]["generated_text"]
        # Return only the assistant tail
        return out.split("[Assistant] ", 1)[-1]

