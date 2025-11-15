import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Путь к корню проекта (где находится pyproject.toml)
PROJECT_ROOT = Path(__file__).parent.parent
# Путь к корню репозитория (на уровень выше live/)
REPO_ROOT = PROJECT_ROOT.parent

def load_prompt(prompt_file_path: str, env_var: str = None) -> str:
    """Загружает промпт из файла или переменной окружения."""
    # Сначала пробуем загрузить из переменной окружения напрямую
    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value
    
    # Если переменной нет, пробуем загрузить из файла
    # Путь может быть относительным (от корня проекта) или абсолютным
    prompt_path = PROJECT_ROOT / prompt_file_path if not os.path.isabs(prompt_file_path) else Path(prompt_file_path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    
    # Если ничего не найдено, возвращаем пустую строку
    return ""

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    MODEL_TEXT = os.getenv("MODEL_TEXT", os.getenv("MODEL"))  # Для обратной совместимости можно использовать MODEL
    MODEL_IMAGE = os.getenv("MODEL_IMAGE")
    SYSTEM_PROMPT_TEXT = load_prompt(
        os.getenv("SYSTEM_PROMPT_TEXT_PATH", "prompts/system_prompt_text.txt"),
        "SYSTEM_PROMPT_TEXT"
    )
    SYSTEM_PROMPT_IMAGE = load_prompt(
        os.getenv("SYSTEM_PROMPT_IMAGE_PATH", "prompts/system_prompt_image.txt"),
        "SYSTEM_PROMPT_IMAGE"
    )
    # Whisper configuration
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
    # SSH configuration for Whisper server
    SSH_HOST = os.getenv("SSH_HOST")
    SSH_USER = os.getenv("SSH_USER", "ubuntu")
    SSH_PORT = int(os.getenv("SSH_PORT", "22"))
    # Путь к SSH ключу (может быть относительным от корня репозитория или абсолютным)
    _ssh_key_file = os.getenv("SSH_KEY_FILE", ".ssh/immers-vm.pem")
    if os.path.isabs(_ssh_key_file):
        SSH_KEY_FILE = _ssh_key_file
    else:
        # Сначала пробуем относительно корня репозитория (где обычно лежит .ssh)
        repo_key_path = REPO_ROOT / _ssh_key_file
        if repo_key_path.exists():
            SSH_KEY_FILE = str(repo_key_path)
        else:
            # Если не найден, пробуем относительно live/
            SSH_KEY_FILE = str(PROJECT_ROOT / _ssh_key_file)

config = Config()

