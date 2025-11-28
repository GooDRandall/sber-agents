# AIDD Bot

Простой Telegram-бот (aiogram 3) с LLM через OpenRouter, файловым контекстом и суммаризацией.

## Требования
- Python 3.12+
- Установленный `uv`

## Установка
```bash
make install
```

## Конфигурация
Создайте локальный `.env` (не коммитьте) со значениями переменных:

```ini
TELEGRAM_BOT_TOKEN=...
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openrouter/auto
LOG_LEVEL=INFO
DATA_DIR=data
WINDOW_SIZE=20
ENABLE_RETRY=false
```

Пояснения:
- `TELEGRAM_BOT_TOKEN` — токен бота.
- `OPENROUTER_API_KEY` — ключ OpenRouter.
- `WINDOW_SIZE` — размер окна истории; суммаризация на кратности окна.

## Запуск
```bash
make run
```

## Команды бота
- `/start` — проверка готовности.
- `/status` — показывает счётчик сообщений, наличие сводки, размер окна.
- `/reset` — очищает историю и сводку текущего чата.

## Структура
- `src/bot/main.py` — входная точка и хендлеры.
- `src/llm/*` — клиент OpenRouter и промпты/суммаризация.
- `src/context.py`, `src/storage.py` — файловый контекст.
- `data/<chat_id>/` — история, сводки и мета.

## Примечания
- Безопасность: ключи/токены только в `.env`, не в репозитории.
- Логи: INFO/ERROR в stdout, без PII.


