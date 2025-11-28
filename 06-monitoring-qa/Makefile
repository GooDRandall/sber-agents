.PHONY: install run dataset dataset-upload ollama-check ollama-serve

install:
	uv sync

run:
	uv run python src/bot.py

dataset:
	uv run python src/dataset_synthesizer.py --create

dataset-upload:
	uv run python src/dataset_synthesizer.py --upload

ollama-check:
	@echo "Проверка Ollama..."
	@ollama list
	@echo ""
	@echo "Проверка доступности сервера..."
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ollama сервер запущен" || echo "❌ Ollama сервер не доступен. Запустите: make ollama-serve"

ollama-serve:
	@echo "Запуск Ollama сервера..."
	@ollama serve

