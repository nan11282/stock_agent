import asyncio
import threading
from types import SimpleNamespace

import telegram_bot
from adapters import LLMTimeoutError
from telegram_bot import (
    _chat_timeout_seconds,
    _delete_message_safely,
    _handle_error,
    _handle_message,
    _run_agent_chat,
    _telegram_connect_timeout_seconds,
    _telegram_connection_pool_size,
    _telegram_pool_timeout_seconds,
)


async def _noop_delete():
    return None


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


def test_telegram_connection_settings_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CONNECTION_POOL_SIZE", "64")
    monkeypatch.setenv("TELEGRAM_POOL_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("TELEGRAM_CONNECT_TIMEOUT_SECONDS", "20")

    assert _telegram_connection_pool_size() == 64
    assert _telegram_pool_timeout_seconds() == 45
    assert _telegram_connect_timeout_seconds() == 20


def test_chat_timeout_default_allows_slow_reasoning(monkeypatch):
    monkeypatch.delenv("TELEGRAM_CHAT_TIMEOUT_SECONDS", raising=False)

    assert _chat_timeout_seconds() == 1800


def test_telegram_connection_settings_fallback_to_safe_minimum(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CONNECTION_POOL_SIZE", "0")
    monkeypatch.setenv("TELEGRAM_POOL_TIMEOUT_SECONDS", "bad")
    monkeypatch.setenv("TELEGRAM_CONNECT_TIMEOUT_SECONDS", "-3")

    assert _telegram_connection_pool_size() == 1
    assert _telegram_pool_timeout_seconds() == 30
    assert _telegram_connect_timeout_seconds() == 1


def test_handle_error_logs_context(capsys):
    update = SimpleNamespace(
        update_id=789,
        effective_chat=SimpleNamespace(id=123),
    )
    context = SimpleNamespace(error=RuntimeError("polling failed"))

    asyncio.run(_handle_error(update, context))

    out = capsys.readouterr().out
    assert "update_id=789" in out
    assert "chat_id=123" in out
    assert "RuntimeError" in out
    assert "polling failed" in out


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
            return SimpleNamespace(delete=_noop_delete)

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


def test_handle_message_sends_thinking_before_agent_init(monkeypatch):
    events = []

    class FakeMessage:
        text = "hello"

        async def reply_text(self, text):
            events.append(f"reply:{text}")
            return SimpleNamespace(delete=_noop_delete)

    class ChatAgent:
        def chat(self, text):
            events.append("chat")
            return "done"

    def get_agent(chat_id):
        events.append("get_agent")
        return ChatAgent()

    message = FakeMessage()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123),
        message=message,
    )

    monkeypatch.setattr(telegram_bot, "_get_agent", get_agent)

    asyncio.run(_handle_message(update, SimpleNamespace()))

    assert events[:2] == ["reply:正在思考...", "get_agent"]
    assert events[-1] == "reply:done"


def test_handle_message_edits_thinking_message_for_stream(monkeypatch):
    class ThinkingMessage:
        def __init__(self):
            self.edits = []
            self.deleted = False

        async def edit_text(self, text):
            self.edits.append(text)

        async def delete(self):
            self.deleted = True

    class FakeMessage:
        text = "hello"

        def __init__(self):
            self.replies = []
            self.thinking = ThinkingMessage()

        async def reply_text(self, text):
            self.replies.append(text)
            if text == "正在思考...":
                return self.thinking
            return SimpleNamespace(delete=_noop_delete)

    class StreamingAgent:
        def chat(self, text, on_text_delta=None):
            on_text_delta("hello ")
            on_text_delta("world")
            return "hello world"

    message = FakeMessage()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123),
        message=message,
    )

    monkeypatch.setattr(telegram_bot, "_get_agent", lambda chat_id: StreamingAgent())

    asyncio.run(_handle_message(update, SimpleNamespace()))

    assert message.replies == ["正在思考..."]
    assert message.thinking.edits[-1] == "hello world"
    assert message.thinking.deleted is False


def test_handle_message_clear_does_not_reset_or_send_thinking(monkeypatch):
    events = []

    class FakeMessage:
        text = "/clear"

        async def reply_text(self, text):
            events.append(f"reply:{text}")
            return SimpleNamespace(delete=_noop_delete)

    class ChatAgent:
        def clear(self):
            events.append("clear")

        def reset(self):
            events.append("reset")

    message = FakeMessage()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123),
        message=message,
    )

    monkeypatch.setattr(telegram_bot, "_get_agent", lambda chat_id: ChatAgent())

    asyncio.run(_handle_message(update, SimpleNamespace()))

    assert events == ["clear", "reply:短期对话已清空（未写入长期记忆）"]


def test_handle_message_replies_when_agent_times_out(monkeypatch):
    class FakeMessage:
        text = "slow question"

        def __init__(self):
            self.replies = []

        async def reply_text(self, text):
            self.replies.append(text)
            return SimpleNamespace(delete=_noop_delete)

    class ChatAgent:
        pass

    async def slow_chat(agent, text):
        await asyncio.sleep(1)
        return "too late"

    message = FakeMessage()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=456),
        message=message,
    )

    monkeypatch.setenv("TELEGRAM_CHAT_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(telegram_bot, "_get_agent", lambda chat_id: ChatAgent())
    monkeypatch.setattr(telegram_bot, "_run_agent_chat", slow_chat)

    asyncio.run(_handle_message(update, SimpleNamespace()))

    assert message.replies[0] == "正在思考..."
    assert message.replies[1].startswith("[分析超时]")


def test_handle_message_reports_llm_timeout_without_raw_exception(monkeypatch):
    class FakeMessage:
        text = "slow question"

        def __init__(self):
            self.replies = []

        async def reply_text(self, text):
            self.replies.append(text)
            return SimpleNamespace(delete=_noop_delete)

    class ChatAgent:
        def chat(self, text):
            raise LLMTimeoutError("LLM request timed out after 180 seconds")

    message = FakeMessage()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=789),
        message=message,
    )

    monkeypatch.setattr(telegram_bot, "_get_agent", lambda chat_id: ChatAgent())

    asyncio.run(_handle_message(update, SimpleNamespace()))

    assert message.replies[0] == "正在思考..."
    assert message.replies[1].startswith("[分析超时]")
    assert "Request timed out" not in message.replies[1]
