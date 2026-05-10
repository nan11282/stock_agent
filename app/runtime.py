"""runtime.py -- 运行时依赖装配

集中构造默认 LLM，避免入口模块在 import 时就把客户端实例化。
"""

from __future__ import annotations

import os

from adapters import OpenAIAdapter


def build_default_llm() -> OpenAIAdapter:
    return OpenAIAdapter(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ.get("DEEPSEEK_API_KEY_stock_agent"),
    )
