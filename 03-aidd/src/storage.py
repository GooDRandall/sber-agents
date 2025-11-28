import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import shutil


def ensure_chat_dir(base_dir: Path, chat_id: int) -> Path:
	chat_dir = base_dir / str(chat_id)
	chat_dir.mkdir(parents=True, exist_ok=True)
	return chat_dir


def append_message(base_dir: Path, chat_id: int, role: str, content: str) -> None:
	chat_dir = ensure_chat_dir(base_dir, chat_id)
	messages_file = chat_dir / "messages.txt"
	ts = int(time.time())
	line = f"{ts}\t{role}\t{content.replace('\n', ' ')}\n"
	with messages_file.open("a", encoding="utf-8") as f:
		f.write(line)
	meta_file = chat_dir / "meta.json"
	meta = {"messages_count": 0}
	if meta_file.exists():
		try:
			meta = json.loads(meta_file.read_text(encoding="utf-8"))
		except Exception:
			meta = {"messages_count": 0}
	meta["messages_count"] = int(meta.get("messages_count", 0)) + 1
	meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def read_last_messages(base_dir: Path, chat_id: int, limit: int) -> List[Dict[str, str]]:
	chat_dir = ensure_chat_dir(base_dir, chat_id)
	messages_file = chat_dir / "messages.txt"
	if not messages_file.exists():
		return []
	lines = messages_file.read_text(encoding="utf-8").splitlines()
	selected = lines[-limit:]
	messages: List[Dict[str, str]] = []
	for line in selected:
		parts = line.split("\t", 2)
		if len(parts) != 3:
			continue
		_, role, content = parts
		messages.append({"role": role, "content": content})
	return messages


def read_summary(base_dir: Path, chat_id: int) -> Optional[str]:
	chat_dir = ensure_chat_dir(base_dir, chat_id)
	summary_file = chat_dir / "summary.txt"
	if not summary_file.exists():
		return None
	text = summary_file.read_text(encoding="utf-8").strip()
	return text or None


def get_messages_count(base_dir: Path, chat_id: int) -> int:
	chat_dir = ensure_chat_dir(base_dir, chat_id)
	meta_file = chat_dir / "meta.json"
	if not meta_file.exists():
		return 0
	try:
		meta = json.loads(meta_file.read_text(encoding="utf-8"))
		return int(meta.get("messages_count", 0))
	except Exception:
		return 0


def write_summary(base_dir: Path, chat_id: int, text: str) -> None:
	chat_dir = ensure_chat_dir(base_dir, chat_id)
	summary_file = chat_dir / "summary.txt"
	summary_file.write_text(text, encoding="utf-8")
	log_file = chat_dir / "summaries.log"
	line = text.replace("\n", " ") + "\n"
	with log_file.open("a", encoding="utf-8") as f:
		f.write(line)


def has_summary(base_dir: Path, chat_id: int) -> bool:
	return read_summary(base_dir, chat_id) is not None


def reset_chat(base_dir: Path, chat_id: int) -> None:
	chat_dir = ensure_chat_dir(base_dir, chat_id)
	# Safely remove files inside chat dir
	for name in ("messages.txt", "summary.txt", "summaries.log", "meta.json"):
		p = chat_dir / name
		if p.exists():
			try:
				p.unlink()
			except Exception:
				pass


