# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 A-share investment assistant with CLI, Telegram bot, scheduled reporting, and persistent memory.

- `app/` contains runtime code: `main.py` for CLI startup, `agent.py` for the ReAct loop, `tools.py` for stock/portfolio tools, `memory.py` for SQLite/ChromaDB retrieval, `scheduler.py` for daily scans, `telegram_bot.py` for chat, and `mailer.py` for reports.
- `tests/` contains pytest tests. Unit tests live in `tests/unit/`; shared setup belongs in `tests/conftest.py`.
- `data/` and `chroma_db/` are local persistent stores mounted into Docker and ignored by git.
- `.env.example` documents required configuration; `.env` holds local secrets and must not be committed.

## Build, Test, and Development Commands

- `python -m venv .venv` then `.venv\Scripts\Activate.ps1`: create and activate a local Windows virtual environment.
- `pip install -r requirements.txt`: install runtime and test dependencies.
- `pytest`: run the full test suite using `pytest.ini` (`pythonpath = app`, quiet output).
- `docker compose build`: build the Python 3.12 application image.
- `docker compose up -d`: start the `agent` and `scheduler` services.
- `docker compose exec -it agent python app/main.py`: open the CLI assistant in the running container.
- `docker compose exec scheduler python app/scheduler.py --now`: trigger a manual scan/report.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, small functions, and descriptive snake_case names for modules, functions, variables, and tests. Keep imports simple and aligned with the flat `app/` module layout. Prefer deterministic helpers for logic that can be unit tested without network, LLM, Telegram, or database access. Comments may be bilingual, but keep them brief.

## Testing Guidelines

Pytest is the test framework. Name tests `test_*.py` and test functions `test_<behavior>`. Put fast, isolated tests in `tests/unit/`; mock or avoid DeepSeek, AKShare, Telegram, SMTP, ChromaDB, and live market APIs unless adding an integration test. Run `pytest` before submitting changes.

## Commit & Pull Request Guidelines

Recent commits use short Chinese summaries, for example `增加单元测试` and `改成在tg上和bot互动...`. Keep subjects concise and focused on one change. Pull requests should include a description, affected commands or services, test results, and configuration changes. Include sample Telegram/CLI output when changing bot behavior or reports.

## Security & Configuration Tips

Never commit `.env`, API keys, SMTP authorization codes, Telegram tokens, `data/`, or `chroma_db/`. When adding configuration, update `.env.example` and `docker-compose.yml` together. Keep proxy settings and timezone behavior explicit because scheduled reports run in `Asia/Shanghai`.
