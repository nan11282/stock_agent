import asyncio
import threading
from types import SimpleNamespace

import telegram_bot
from telegram_bot import _delete_message_safely, _handle_message, _run_agent_chat


class FakeAgent:
    def __init__(self):
        self.worker_thread = None

    def chat(self, text):
        self.worker_thread = threading.get_ident()
        return f"echo: {text}"


def test_run_agent_chat_uses_background_thread():
    agent = FakeAgent()
    caller_thread = threading.get_ident()

    out = asyncio.run(_run_agent_chat(agent, "hello"))

    assert out == "echo: hello"
    assert agent.worker_thread is not None
    assert agent.worker_thread != caller_thread


class FakeTelegramError(Exception):
    pass


def test_delete_message_safely_ignores_telegram_timeout(monkeypatch):
    class TimeoutMessage:
        async def delete(self):
            raise FakeTelegramError("delete timed out")

    monkeypatch.setattr(telegram_bot, "TelegramError", FakeTelegramError)
    asyncio.run(_delete_message_safely(TimeoutMessage()))


def test_handle_message_sends_response_when_thinking_delete_times_out(monkeypatch):
    class TimeoutThinkingMessage:
        async def delete(self):
            raise FakeTelegramError("delete timed out")

    class FakeMessage:
        text = "hello"

        def __init__(self):
            self.replies = []

        async def reply_text(self, text):
            self.replies.append(text)
            if text == "正在思考...":
                return TimeoutThinkingMessage()
            return SimpleNamespace(delete=lambda: None)

    class ChatAgent:
        def chat(self, text):
            return f"echo: {text}"

    message = FakeMessage()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123),
        message=message,
    )

    monkeypatch.setattr(telegram_bot, "_get_agent", lambda chat_id: ChatAgent())
    monkeypatch.setattr(telegram_bot, "TelegramError", FakeTelegramError)

    asyncio.run(_handle_message(update, SimpleNamespace()))

    assert message.replies == ["正在思考...", "echo: hello"]
