"""
telegram_bot.py — Telegram 聊天助理

每条用户消息交给独立的 Agent 实例（ReAct + ALL_TOOLS），
LLM 自主调工具或闲聊，回复自动分段适配 Telegram 长度限制。
"""

import asyncio
import os

from agent import Agent
from runtime import build_default_llm

try:
    from telegram.error import TelegramError
except ImportError:
    TelegramError = Exception

# LLM 适配器——和 main.py 同一套，按需构造
_llm = None

# per-user Agent 实例池
_agents: dict[int, Agent] = {}
_agent_locks: dict[int, asyncio.Lock] = {}


def _get_llm():
    global _llm
    if _llm is None:
        _llm = build_default_llm()
    return _llm


def _get_agent(chat_id: int) -> Agent:
    if chat_id not in _agents:
        _agents[chat_id] = Agent(llm=_get_llm(), max_steps=20)
    return _agents[chat_id]


def _get_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _agent_locks:
        _agent_locks[chat_id] = asyncio.Lock()
    return _agent_locks[chat_id]


async def _run_agent_chat(agent: Agent, text: str) -> str:
    return await asyncio.to_thread(agent.chat, text)


async def _delete_message_safely(message) -> None:
    """Best-effort cleanup for transient Telegram API/network failures."""
    try:
        await message.delete()
    except TelegramError as e:
        print(f"[TelegramBot] 删除思考提示失败，已忽略: {e}")


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

    agent = _get_agent(chat_id)

    # /reset 清空对话
    if text.strip().lower() in ("/reset", "reset"):
        agent.reset()
        await update.message.reply_text("对话已清空（记忆和数据库保留）")
        return

    # 发送思考提示
    thinking_msg = await update.message.reply_text("正在思考...")

    try:
        async with _get_lock(chat_id):
            response = await _run_agent_chat(agent, text)
    except Exception as e:
        response = f"[分析出错] {e}"

    # 删除思考提示，分段发送回复
    await _delete_message_safely(thinking_msg)
    for part in _split_long_message(response):
        await update.message.reply_text(part)


def start_bot():
    """启动 Telegram bot（阻塞当前线程，用事件驱动长轮询）。"""

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("[TelegramBot] 未设置 TELEGRAM_BOT_TOKEN，跳过启动")
        return

    from telegram.ext import Application, MessageHandler, filters

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, _handle_message
    ))
    app.add_handler(MessageHandler(filters.COMMAND, _handle_message))
    print("[TelegramBot] 已启动 — 事件驱动长轮询")
    app.run_polling()
