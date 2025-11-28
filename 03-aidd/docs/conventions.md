# Conventions for Code Generation (KISS/YAGNI)

- Focus on MVP only; implement the minimum to satisfy behavior. No over‑engineering.
- Follow `docs/vision.md` for system behavior and boundaries. Do not redefine specs here.

## General
- Python 3.12. Use standard library where possible; add deps only if essential.
- Keep modules small, readable, and cohesive. Prefer explicit code over indirection.
- Avoid deep nesting; use guard clauses and clear early returns.
- Handle errors minimally: log and fail safe; do not suppress silently.
- No secrets or tokens in code or VCS. Read from environment only.

## Dependencies & Build
- Dependency manager: uv. Lock and install via Makefile targets.
- Make targets (only): `install`, `run`, `clean`. Keep them simple and portable.

## Structure (high level)
- Respect the layout from `docs/vision.md` (`src/bot`, `src/llm`, `src/*.py`, `data/`).
- Keep Telegram logic in `bot`, LLM calls in `llm`, infra in `src` root.

## Telegram Bot
- aiogram 3.x with polling only. No webhooks.
- Single-message replies (no streaming). Keep handlers simple.

## LLM Usage
- Use official `openai` client configured for OpenRouter (see `vision.md`).
- System prompt + (summary if exists) + last window of messages + user input.
- Add minimal timeouts; optional single retry on network errors.

## Context & Storage
- Store chat context in plain text files under `data/<chat_id>/`.
- Maintain a rolling window of 20 messages.
- After each 20 messages, create/update a concise Russian summary; merge summaries sequentially.

## Configuration
- Load config once at startup from environment (`.env` allowed). No runtime mutation.
- Use defaults defined in `vision.md`. Do not hardcode paths or tokens.

## Logging
- Use standard `logging`. Levels: INFO/ERROR to stdout.
- Log metadata (model, duration, status), errors, and summary events. Avoid logging PII/content by default.

## Code Style
- Prefer clarity over cleverness. Name things descriptively.
- Keep files short; split only when it improves readability.
- Comments only for non-obvious rationale; avoid redundant comments.

## Testing & Delivery
- MVP: manual testing via Telegram. No CI/CD, no mandatory linters/formatters.
- Document run instructions in README succinctly and keep `docs/vision.md` up to date.


