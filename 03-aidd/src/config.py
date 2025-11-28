import os
from pathlib import Path


def load_env_from_dotenv(dotenv_path: Path = Path(".env")) -> None:
	if not dotenv_path.exists():
		return
	for line in dotenv_path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key, value = key.strip(), value.strip().strip('"').strip("'")
		os.environ.setdefault(key, value)


class Config:
	def __init__(self) -> None:
		load_env_from_dotenv()
		self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
		self.log_level = os.environ.get("LOG_LEVEL", "INFO")
		# OpenRouter / LLM settings
		self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
		self.openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
		self.openrouter_model = os.environ.get("OPENROUTER_MODEL", "openrouter/auto")
		self.enable_retry = os.environ.get("ENABLE_RETRY", "false").lower() == "true"
		# Context storage
		self.data_dir = os.environ.get("DATA_DIR", "data")
		self.window_size = int(os.environ.get("WINDOW_SIZE", "20"))


