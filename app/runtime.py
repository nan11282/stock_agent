"""runtime.py -- 运行时依赖装配

集中构造默认 LLM，避免入口模块在 import 时就把客户端实例化。
"""

from __future__ import annotations

import os

from adapters import OpenAIAdapter


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def build_default_llm() -> OpenAIAdapter:
    # 项目默认用 OpenAI 兼容协议接 DeepSeek。
    # 所有入口都从这里拿 LLM，保证 CLI、Telegram、scheduler 的模型和超时策略一致。
    return OpenAIAdapter(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ.get("DEEPSEEK_API_KEY_stock_agent"),
        timeout=_env_float("LLM_TIMEOUT_SECONDS", 180.0),
        max_retries=max(0, _env_int("LLM_MAX_RETRIES", 0)),
    )
