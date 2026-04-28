"""
telegram_bot.py — Telegram 聊天助理

每条用户消息交给独立的 Agent 实例（ReAct + ALL_TOOLS），
LLM 自主调工具或闲聊，回复自动分段适配 Telegram 长度限制。
"""

import os

from adapters import OpenAIAdapter
from agent import Agent

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# LLM 适配器——和 main.py 同一套
_llm = OpenAIAdapter(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY_stock_agent"),
)

# per-user Agent 实例池
_agents: dict[int, Agent] = {}


def _get_agent(chat_id: int) -> Agent:
    if chat_id not in _agents:
        _agents[chat_id] = Agent(llm=_llm, max_steps=20)
    return _agents[chat_id]


def _split_long_message(text: str, limit: int = 4000) -> list[str]:
    """Telegram 消息上限 4096，按段落分行后拼到 limit 以内。"""
    parts = []
    current = ""
    for paragraph in text.split("\n"):
        if len(current) + len(paragraph) + 1 > limit:
            if current:
                parts.append(current.strip())
            current = paragraph
        else:
            current += "\n" + paragraph if current else paragraph
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
        response = agent.chat(text)
    except Exception as e:
        response = f"[分析出错] {e}"

    # 删除思考提示，分段发送回复
    await thinking_msg.delete()
    for part in _split_long_message(response):
        await update.message.reply_text(part)


def start_bot():
    """启动 Telegram bot（阻塞当前线程，用事件驱动长轮询）。"""

    if not TELEGRAM_TOKEN:
        print("[TelegramBot] 未设置 TELEGRAM_BOT_TOKEN，跳过启动")
        return

    from telegram.ext import Application, MessageHandler, filters

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, _handle_message
    ))
    app.add_handler(MessageHandler(filters.COMMAND, _handle_message))
    print("[TelegramBot] 已启动 — 事件驱动长轮询")
    app.run_polling()
