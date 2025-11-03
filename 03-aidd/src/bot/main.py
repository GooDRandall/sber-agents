import asyncio
from pathlib import Path
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command

from src.config import Config
from src.logging_setup import setup_logging
from src.llm.client import LLMClient
from src.llm.prompt import build_messages
from src.context import ChatContext


async def main() -> None:
	cfg = Config()
	setup_logging(cfg.log_level)
	bot = Bot(token=cfg.telegram_bot_token)
	dp = Dispatcher()

	@dp.message(CommandStart())
	async def start(message: types.Message) -> None:
		await message.answer("Бот готов к работе")

	@dp.message(Command("status"))
	async def status(message: types.Message) -> None:
		ctx = ChatContext(cfg, chat_id=message.chat.id)
		from src import storage
		count = storage.get_messages_count(Path(cfg.data_dir), message.chat.id)
		has_sum = storage.has_summary(Path(cfg.data_dir), message.chat.id)
		text = f"Сообщений: {count}. Сводка: {'есть' if has_sum else 'нет'}. Окно: {cfg.window_size}."
		await message.answer(text)

	@dp.message(Command("reset"))
	async def reset(message: types.Message) -> None:
		from src import storage
		storage.reset_chat(Path(cfg.data_dir), message.chat.id)
		await message.answer("История и сводка очищены.")

	@dp.message()
	async def handle_text(message: types.Message) -> None:
		if not message.text or message.text.startswith("/"):
			return
		ctx = ChatContext(cfg, chat_id=message.chat.id)
		history = ctx.get_history_window()
		summary = ctx.get_summary()
		client = LLMClient(cfg)
		msgs = build_messages(message.text, summary=summary, history=history)
		try:
			ctx.append_user(message.text)
			reply = client.generate(msgs)
			ctx.append_assistant(reply or "")
			await message.answer(reply or "Не удалось получить ответ.")
			# attempt summarization across restarts using file counters
			ctx.maybe_summarize_with_llm(client)
		except Exception:
			await message.answer("Сервис временно недоступен, попробуйте позже.")

	await dp.start_polling(bot)


if __name__ == "__main__":
	asyncio.run(main())


