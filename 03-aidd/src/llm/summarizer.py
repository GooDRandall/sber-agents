from typing import List, Dict, Optional

from src.llm.client import LLMClient


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
	parts: List[str] = []
	for m in messages:
		role = m.get("role", "")
		content = m.get("content", "")
		parts.append(f"{role}: {content}")
	return "\n".join(parts)


def summarize_block(client: LLMClient, previous_summary: Optional[str], block_messages: List[Dict[str, str]]) -> str:
	block_text = _messages_to_text(block_messages)
	instruction = (
		"Сожми диалог (последний блок сообщений) в краткую русскую сводку 5–8 строк. "
		"Если есть предыдущая сводка, объедини их в одну актуальную сводку. "
		"Пиши короткими предложениями, без маркеров списков."
	)
	messages = [
		{"role": "system", "content": instruction},
	]
	if previous_summary:
		messages.append({"role": "system", "content": f"Предыдущая сводка:\n{previous_summary}"})
	messages.append({"role": "user", "content": f"Диалог:\n{block_text}"})
	return client.generate(messages)


