# backends_gemini.py
import os
import time
from typing import List

# Google Gemini SDK
try:
    import google.generativeai as genai
    from google.api_core.exceptions import (
        ResourceExhausted,
        RetryError,
        ServiceUnavailable,
        DeadlineExceeded,
    )
except ImportError as e:
    genai = None
    # define fallbacks so import-time references don't explode; we still raise in __init__
    ResourceExhausted = RetryError = ServiceUnavailable = DeadlineExceeded = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backends import BaseLLMClient, LLMConfig, Message


class GeminiClient(BaseLLMClient):
    """
    Google Gemini backend with:
      - Exponential backoff retries for transient 429/5xx/timeouts
      - Fast-fail on daily quota errors (no point retrying)
      - Optional pacing between successful calls via GEMINI_SLEEP
    """

    def __init__(self):
        if genai is None:
            raise RuntimeError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for MODEL_PROVIDER=gemini")
        genai.configure(api_key=api_key)

        # Backoff & pacing knobs (env-driven)
        self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
        self.backoff_min = float(os.getenv("GEMINI_BACKOFF_MIN", "2.0"))   # seconds
        self.backoff_max = float(os.getenv("GEMINI_BACKOFF_MAX", "30.0"))  # seconds
        self.sleep_s     = float(os.getenv("GEMINI_SLEEP", "1.0"))         # seconds between successful calls

    def _retry_decorator(self):
        """
        Build a retry decorator using instance-configured knobs.
        Retries ResourceExhausted (per-minute), ServiceUnavailable, DeadlineExceeded, RetryError.
        """
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=self.backoff_min, max=self.backoff_max),
            retry=retry_if_exception_type((
                ResourceExhausted,   # rate/quotas (per-minute)
                ServiceUnavailable,  # transient backend
                DeadlineExceeded,    # timeouts
                RetryError           # generic Google retry wrapper
            )),
        )

    def _format_chat_as_prompt(self, messages: List[Message]) -> str:
        """
        Convert OpenAI-style messages into a single prompt for Gemini.generate_content.
        """
        sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = []
        for m in messages:
            if m["role"] == "user":
                convo.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                convo.append(f"Assistant: {m['content']}")
        return ((sys + "\n\n") if sys else "") + "\n".join(convo)

    def chat(self, messages: List[Message], cfg: LLMConfig) -> str:
        model = genai.GenerativeModel(cfg.model)
        prompt = self._format_chat_as_prompt(messages)

        retry_deco = self._retry_decorator()

        @retry_deco
        def _call():
            # We wrap the call so Tenacity can retry *only* transient errors.
            try:
                return model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=cfg.temperature,
                        max_output_tokens=cfg.max_tokens,
                    ),
                )
            except Exception as e:
                # If it's clearly a DAILY quota error, don't waste retries.
                # Common substrings in Gemini 429s for daily caps:
                # "RequestsPerDay", "PerDay", "GenerateRequestsPerDayPerProjectPerModel"
                msg = str(e)
                if ("RequestsPerDay" in msg) or ("PerDay" in msg) or ("GenerateRequestsPerDayPerProjectPerModel" in msg):
                    # Re-raise as-is; Tenacity will not classify it as retryable unless it matches retry_if_exception_type.
                    raise
                # Otherwise, let Tenacity see it; if it's in retry set, it will backoff and retry.
                raise

        response = _call()

        # Optional pacing after a successful response
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)

        # Return plain text
        return (getattr(response, "text", None) or "").strip()
