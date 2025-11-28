import logging
import time
from typing import List, Dict, Any

from openai import OpenAI

from src.config import Config


logger = logging.getLogger(__name__)


class LLMClient:
	def __init__(self, cfg: Config) -> None:
		self._client = OpenAI(api_key=cfg.openrouter_api_key, base_url=cfg.openrouter_base_url)
		self._model = cfg.openrouter_model
		self._enable_retry = cfg.enable_retry

	def generate(self, messages: List[Dict[str, Any]], timeout_seconds: int = 30) -> str:
		start_ts = time.time()
		attempts = 2 if self._enable_retry else 1
		last_exc: Exception | None = None
		for attempt in range(1, attempts + 1):
			try:
				response = self._client.chat.completions.create(
					model=self._model,
					messages=messages,
					temperature=0.7,
					timeout=timeout_seconds,
				)
				text = response.choices[0].message.content or ""
				duration_ms = int((time.time() - start_ts) * 1000)
				logger.info("llm_call model=%s duration_ms=%s status=ok", self._model, duration_ms)
				return text
			except Exception as exc:
				last_exc = exc
				if attempt < attempts:
					logger.error("llm_call_failed attempt=%s retrying error=%s", attempt, type(exc).__name__)
					continue
				duration_ms = int((time.time() - start_ts) * 1000)
				logger.error("llm_call model=%s duration_ms=%s status=error error=%s", self._model, duration_ms, repr(exc))
				raise


