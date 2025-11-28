from pathlib import Path
from typing import List, Dict, Optional

from src.config import Config
from src import storage
from src.llm.summarizer import summarize_block
from src.llm.client import LLMClient


class ChatContext:
	def __init__(self, cfg: Config, chat_id: int) -> None:
		self._base_dir = Path(cfg.data_dir)
		self._window = cfg.window_size
		self._chat_id = chat_id

	def get_history_window(self) -> List[Dict[str, str]]:
		return storage.read_last_messages(self._base_dir, self._chat_id, self._window)

	def get_summary(self) -> Optional[str]:
		return storage.read_summary(self._base_dir, self._chat_id)

	def append_user(self, text: str) -> None:
		storage.append_message(self._base_dir, self._chat_id, "user", text)

	def append_assistant(self, text: str) -> None:
		storage.append_message(self._base_dir, self._chat_id, "assistant", text)

	def maybe_summarize_with_llm(self, client: LLMClient) -> bool:
		count = storage.get_messages_count(self._base_dir, self._chat_id)
		if count <= 0 or self._window <= 0:
			return False
		# Суммаризация происходит на кратности окна
		if count % self._window != 0:
			return False
		block = self.get_history_window()
		prev = self.get_summary()
		try:
			new_summary = summarize_block(client, prev, block)
			if new_summary:
				storage.write_summary(self._base_dir, self._chat_id, new_summary)
				return True
		except Exception:
			return False
		return False


