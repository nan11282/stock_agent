"""
telegram_bot.py — Telegram 聊天助理

每条用户消息交给独立的 Agent 实例（ReAct + ALL_TOOLS），
LLM 自主调工具或闲聊，回复自动分段适配 Telegram 长度限制。
"""

import asyncio
import os
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


def _chat_timeout_seconds() -> int:
    return max(1, _env_int("TELEGRAM_CHAT_TIMEOUT_SECONDS", 900))


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


async def _run_agent_chat(agent: Agent, text: str) -> str:
    # Agent.chat 是同步的，里面会访问行情、SQLite、ChromaDB 和 LLM；
    # 放到线程里执行，避免阻塞 python-telegram-bot 的事件循环。
    return await asyncio.to_thread(agent.chat, text)


async def _delete_message_safely(message) -> None:
    """Best-effort cleanup for transient Telegram API/network failures."""
    try:
        await message.delete()
    except TelegramError as e:
        print(f"[TelegramBot] 删除思考提示失败，已忽略: {e}")


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


async def _handle_message(update, context):
    """Telegram 消息入口。"""
    chat_id = update.effective_chat.id
    text = update.message.text or ""

    if not text.strip():
        return

    print(f"[TelegramBot] 收到消息 chat_id={chat_id} chars={len(text)}")

    # /reset 清空对话
    if text.strip().lower() in ("/reset", "reset"):
        # reset 只清短期会话，不清持仓、自选、决策日志和长期洞察。
        agent = _get_agent(chat_id)
        agent.reset()
        await update.message.reply_text("对话已清空（记忆和数据库保留）")
        return

    # 先发送思考提示，再做 Agent/DB/Chroma 初始化，避免用户端无反馈。
    thinking_msg = None
    try:
        thinking_msg = await update.message.reply_text("正在思考...")
        print(f"[TelegramBot] 已发送思考提示 chat_id={chat_id}")
    except TelegramError as e:
        print(f"[TelegramBot] 发送思考提示失败 chat_id={chat_id}: {e}")

    agent = _get_agent(chat_id)

    try:
        timeout = _chat_timeout_seconds()
        async with _get_lock(chat_id):
            # 同一个用户的消息串行处理，避免“确认写入”和新问题交错导致状态错乱。
            response = await asyncio.wait_for(_run_agent_chat(agent, text), timeout=timeout)
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

    # 删除思考提示，分段发送回复
    if thinking_msg is not None:
        await _delete_message_safely(thinking_msg)
    for part in _split_long_message(response):
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
