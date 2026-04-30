"""
conftest.py — 全局测试配置

负责：
- 在 import 业务模块之前塞好 fake API key（OpenAI/DeepSeek client 构造时会
  从 env 取 key，缺失会 raise）
- 提供常用 fixture（临时 DB 等）
"""

import os
import sys

# ── 必须最早执行：避免 import telegram_bot / scheduler 时
# OpenAI() 构造抛 "no API key" ──
os.environ.setdefault("DEEPSEEK_API_KEY_stock_agent", "test-fake-key")
os.environ.setdefault("OPENAI_API_KEY", "test-fake-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-fake-key")
os.environ.setdefault("TRACE_ENABLED", "0")

# 防止测试误写到项目真实的 ./data /chroma_db
os.environ.setdefault("DB_PATH", ":memory:")

import pytest


@pytest.fixture
def fresh_decisionlog():
    """每个测试用例独立的 in-memory SQLite DecisionLog。"""
    from memory import DecisionLog
    log = DecisionLog(db_path=":memory:")
    yield log
    log.conn.close()


@pytest.fixture
def tmp_chroma(tmp_path):
    """临时 ChromaDB 目录（避免污染真实 ./chroma_db）。"""
    return str(tmp_path / "chroma")
