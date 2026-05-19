"""
telegram_bot.py — Telegram 聊天助理

每条用户消息交给独立的 Agent 实例（ReAct + ALL_TOOLS），
LLM 自主调工具或闲聊，回复自动分段适配 Telegram 长度限制。
"""

import asyncio
import inspect
import os
import threading
import time
import traceback

from agent import Agent
from adapters import LLMTimeoutError
from runtime import build_default_llm

try:
    from telegram.error import TelegramError
except ImportError:
    TelegramError = Exception

# LLM 适配器——和 main.py 同一套，按需构造
_llm = None

# per-user Agent 实例池
# Telegram 是多用户入口：每个 chat_id 拥有独立 Agent 和短期对话历史，
# 但共享同一套长期记忆/决策数据库，符合“多人入口、一个投资账本”的业务模型。
_agents: dict[int, Agent] = {}
_agent_locks: dict[int, asyncio.Lock] = {}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _chat_timeout_seconds() -> int:
    return max(1, _env_int("TELEGRAM_CHAT_TIMEOUT_SECONDS", 1800))


def _telegram_connection_pool_size() -> int:
    return max(1, _env_int("TELEGRAM_CONNECTION_POOL_SIZE", 32))


def _telegram_pool_timeout_seconds() -> int:
    return max(1, _env_int("TELEGRAM_POOL_TIMEOUT_SECONDS", 30))


def _telegram_connect_timeout_seconds() -> int:
    return max(1, _env_int("TELEGRAM_CONNECT_TIMEOUT_SECONDS", 30))


def _telegram_read_timeout_seconds() -> int:
    return max(1, _env_int("TELEGRAM_READ_TIMEOUT_SECONDS", 60))


def _telegram_get_updates_pool_size() -> int:
    return max(1, _env_int("TELEGRAM_GET_UPDATES_POOL_SIZE", 8))


def _telegram_stream_edit_interval_seconds() -> float:
    return max(0.2, _env_float("TELEGRAM_STREAM_EDIT_INTERVAL_SECONDS", 1.0))


def _telegram_stream_min_chars() -> int:
    return max(20, _env_int("TELEGRAM_STREAM_MIN_CHARS", 80))


def _get_llm():
    global _llm
    if _llm is None:
        # LLM 客户端延迟初始化，避免导入 bot 模块时就要求环境变量/网络都可用。
        _llm = build_default_llm()
    return _llm


def _get_agent(chat_id: int) -> Agent:
    if chat_id not in _agents:
        _agents[chat_id] = Agent(
            llm=_get_llm(),
            max_steps=max(1, _env_int("TELEGRAM_AGENT_MAX_STEPS", 8)),
        )
    return _agents[chat_id]


def _get_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _agent_locks:
        _agent_locks[chat_id] = asyncio.Lock()
    return _agent_locks[chat_id]


def _discard_chat_state(chat_id: int) -> None:
    _agents.pop(chat_id, None)
    _agent_locks.pop(chat_id, None)


def _agent_accepts_stream_callback(agent: Agent) -> bool:
    try:
        return "on_text_delta" in inspect.signature(agent.chat).parameters
    except (TypeError, ValueError):
        return False


def _call_agent_chat(agent: Agent, text: str, on_text_delta=None) -> str:
    if on_text_delta is not None and _agent_accepts_stream_callback(agent):
        return agent.chat(text, on_text_delta=on_text_delta)
    return agent.chat(text)


async def _run_agent_chat(agent: Agent, text: str, on_delta=None) -> str:
    # Agent.chat 是同步的，里面会访问行情、SQLite、ChromaDB 和 LLM；
    # 放到线程里执行，避免阻塞 python-telegram-bot 的事件循环。
    if on_delta is None:
        return await asyncio.to_thread(_call_agent_chat, agent, text, None)

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    stopped = threading.Event()

    def put_event(kind: str, payload):
        if stopped.is_set():
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
        except RuntimeError:
            pass

    def emit(delta: str):
        if delta:
            put_event("delta", delta)

    def run():
        try:
            response = _call_agent_chat(agent, text, emit)
            put_event("done", response)
        except BaseException as e:
            put_event("error", e)

    worker = asyncio.create_task(asyncio.to_thread(run))
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "delta":
                await on_delta(payload)
                continue
            if kind == "error":
                raise payload
            await worker
            return payload
    finally:
        if not worker.done():
            stopped.set()
            worker.cancel()


async def _delete_message_safely(message) -> None:
    """Best-effort cleanup for transient Telegram API/network failures."""
    try:
        await message.delete()
    except TelegramError as e:
        print(f"[TelegramBot] 删除思考提示失败，已忽略: {e}")


async def _edit_message_safely(message, text: str) -> bool:
    edit_text = getattr(message, "edit_text", None)
    if edit_text is None:
        return False
    try:
        await edit_text(text)
        return True
    except TelegramError as e:
        print(f"[TelegramBot] 编辑流式回复失败，已忽略: {e}")
        return False


async def _handle_error(update, context) -> None:
    """Log Telegram polling/handler failures with enough context to diagnose."""
    error = getattr(context, "error", None)
    update_id = getattr(update, "update_id", None) if update is not None else None
    chat_id = getattr(getattr(update, "effective_chat", None), "id", None)
    print(
        "[TelegramBot] 异常"
        f" update_id={update_id}"
        f" chat_id={chat_id}"
        f" type={type(error).__name__}"
        f" error={error}"
    )
    if error is not None:
        print("".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip())


def _split_long_message(text: str, limit: int = 4000) -> list[str]:
    """Telegram 消息上限 4096，按段落分行后拼到 limit 以内。"""
    parts = []
    current = ""
    for paragraph in text.split("\n"):
        chunks = [paragraph[i:i + limit] for i in range(0, len(paragraph), limit)] or [""]
        for chunk in chunks:
            if not current:
                current = chunk
                continue
            if len(current) + len(chunk) + 1 > limit:
                parts.append(current.strip())
                current = chunk
            else:
                current += "\n" + chunk
    if current.strip():
        parts.append(current.strip())
    return parts or [text[:limit]]


def _stream_preview(text: str, limit: int = 3900) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit - 12].rstrip() + "\n...[输出中]"


async def _handle_message(update, context):
    """Telegram 消息入口。"""
    chat_id = update.effective_chat.id
    text = update.message.text or ""

    if not text.strip():
        return

    print(f"[TelegramBot] 收到消息 chat_id={chat_id} chars={len(text)}")

    # /reset 结束会话并沉淀摘要；/clear 只清短期上下文。
    if text.strip().lower() in ("/reset", "reset"):
        # reset 会先把当前短期会话摘要写入长期记忆，再清空短期上下文。
        agent = _get_agent(chat_id)
        agent.reset()
        await update.message.reply_text("对话已清空，并已沉淀会话摘要（数据库和持仓记录保留）")
        return
    if text.strip().lower() in ("/clear", "clear"):
        # clear 是纯粹的上下文丢弃：不写长期记忆，不改持仓/自选/决策库。
        agent = _get_agent(chat_id)
        agent.clear()
        await update.message.reply_text("短期对话已清空（未写入长期记忆）")
        return

    # 先发送思考提示，再做 Agent/DB/Chroma 初始化，避免用户端无反馈。
    thinking_msg = None
    try:
        thinking_msg = await update.message.reply_text("正在思考...")
        print(f"[TelegramBot] 已发送思考提示 chat_id={chat_id}")
    except TelegramError as e:
        print(f"[TelegramBot] 发送思考提示失败 chat_id={chat_id}: {e}")

    agent = _get_agent(chat_id)
    streamed_parts = []
    last_stream_edit_at = 0.0
    last_stream_preview = ""

    async def publish_delta(delta: str):
        nonlocal last_stream_edit_at, last_stream_preview
        streamed_parts.append(delta)
        if thinking_msg is None:
            return

        preview = _stream_preview("".join(streamed_parts))
        if not preview or preview == last_stream_preview:
            return

        now = time.monotonic()
        enough_time = now - last_stream_edit_at >= _telegram_stream_edit_interval_seconds()
        enough_chars = len(preview) - len(last_stream_preview) >= _telegram_stream_min_chars()
        if last_stream_preview and not (enough_time or enough_chars):
            return

        if await _edit_message_safely(thinking_msg, preview):
            last_stream_edit_at = now
            last_stream_preview = preview

    try:
        timeout = _chat_timeout_seconds()
        async with _get_lock(chat_id):
            # 同一个用户的消息串行处理，避免“确认写入”和新问题交错导致状态错乱。
            try:
                chat_task = _run_agent_chat(agent, text, on_delta=publish_delta)
            except TypeError:
                chat_task = _run_agent_chat(agent, text)
            response = await asyncio.wait_for(
                chat_task,
                timeout=timeout,
            )
    except asyncio.TimeoutError:
        # 超时后丢弃该 chat 的短期 Agent 状态，防止下一轮继续沿用半完成的工具/确认上下文。
        _discard_chat_state(chat_id)
        response = (
            f"[分析超时] 本轮处理超过 {timeout} 秒，已停止等待并重置本轮会话。"
            "可以把问题拆短一点，或稍后再试。"
        )
    except LLMTimeoutError as e:
        print(f"[TelegramBot] LLM 超时 chat_id={chat_id}: {e}")
        response = (
            "[分析超时] 上游模型响应超过当前 LLM_TIMEOUT_SECONDS 限制。"
            "这不是 Telegram 发送失败；可以稍后重试，或把 LLM_TIMEOUT_SECONDS 调大。"
        )
    except Exception as e:
        print(f"[TelegramBot] 处理消息失败 chat_id={chat_id}: {e}")
        response = f"[分析出错] {e}"

    # 删除/复用思考提示，分段发送回复。流式时优先把首段落编辑到原消息里，避免重复刷屏。
    first_part_already_sent = False
    if thinking_msg is not None:
        streamed_text = "".join(streamed_parts).strip()
        parts = _split_long_message(response)
        if streamed_text and streamed_text == (response or "").strip():
            first_part_already_sent = await _edit_message_safely(thinking_msg, parts[0])
            if first_part_already_sent:
                print(f"[TelegramBot] 已编辑最终首段 chat_id={chat_id} chars={len(parts[0])}")
        if not first_part_already_sent:
            await _delete_message_safely(thinking_msg)

    parts = _split_long_message(response)
    for part in parts[1 if first_part_already_sent else 0:]:
        try:
            await update.message.reply_text(part)
            print(f"[TelegramBot] 已发送回复 chat_id={chat_id} chars={len(part)}")
        except TelegramError as e:
            print(f"[TelegramBot] 发送回复失败 chat_id={chat_id}: {e}")
            break


def start_bot():
    """启动 Telegram bot（阻塞当前线程，用事件驱动长轮询）。"""

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        # scheduler 容器也会调用 start_bot；未配置 token 时允许只跑定时扫描。
        print("[TelegramBot] 未设置 TELEGRAM_BOT_TOKEN，跳过启动")
        return

    from telegram.ext import Application, MessageHandler, filters

    pool_size = _telegram_connection_pool_size()
    pool_timeout = _telegram_pool_timeout_seconds()
    connect_timeout = _telegram_connect_timeout_seconds()
    read_timeout = _telegram_read_timeout_seconds()
    get_updates_pool_size = _telegram_get_updates_pool_size()
    chat_timeout = _chat_timeout_seconds()

    builder = (
        Application.builder()
        .token(token)
        .connection_pool_size(pool_size)
        .pool_timeout(pool_timeout)
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .write_timeout(connect_timeout)
        .get_updates_connection_pool_size(get_updates_pool_size)
        .get_updates_pool_timeout(pool_timeout)
        .get_updates_connect_timeout(connect_timeout)
        .get_updates_read_timeout(read_timeout)
        .get_updates_write_timeout(connect_timeout)
    )
    app = builder.build()
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, _handle_message
    ))
    app.add_handler(MessageHandler(filters.COMMAND, _handle_message))
    app.add_error_handler(_handle_error)
    print(
        "[TelegramBot] 已启动 — 事件驱动长轮询 "
        f"connection_pool_size={pool_size} "
        f"get_updates_pool_size={get_updates_pool_size} "
        f"chat_timeout={chat_timeout}s "
        f"pool_timeout={pool_timeout}s "
        f"connect_timeout={connect_timeout}s "
        f"read_timeout={read_timeout}s"
    )
    app.run_polling(
        poll_interval=1.0,
        timeout=30,
        bootstrap_retries=-1,
        drop_pending_updates=False,
    )
