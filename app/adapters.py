"""
adapters.py — LLM 适配层
Agent 只和中性 Message 类型打交道；每个 Adapter 负责把中性格式
转换成自家 API 所需的形状，再把 API 响应转回中性。
切换模型 = 换一个 Adapter 实例，Agent 代码零改动。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# 中性内部格式（Provider-agnostic）
# ─────────────────────────────────────────────

@dataclass
class ToolCall:
    # LLM 说“要调用哪个业务工具、带什么参数”，Agent 再决定是否允许执行。
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    tool_call_id: str
    content: str


@dataclass
class Message:
    """Agent 历史中的一条消息。
    - user 的普通消息：只填 text
    - user 的工具结果回合：只填 tool_results
    - assistant：填 text 和/或 tool_calls
    """
    role: str  # "user" | "assistant"
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    reasoning_content: str | None = None  # DeepSeek V4 推理链


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    reasoning_content: str | None = None


# ─────────────────────────────────────────────
# Adapter 接口
# ─────────────────────────────────────────────

class LLMAdapter(ABC):
    # Agent 只依赖这个抽象接口，因此业务循环不关心底层是 Claude、OpenAI 还是 DeepSeek。
    @abstractmethod
    def chat(self, messages: list[Message], tools: list[dict], system: str) -> LLMResponse:
        ...


# ─────────────────────────────────────────────
# Claude Adapter
# ─────────────────────────────────────────────

class ClaudeAdapter(LLMAdapter):
    def __init__(self, model: str = "claude-opus-4-7"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    @staticmethod
    def _to_anthropic(messages: list[Message]) -> list[dict]:
        # Anthropic 把 tool_result 放在 user content block 里，
        # 适配层负责格式差异，避免污染 Agent 的业务逻辑。
        out: list[dict] = []
        for m in messages:
            if m.role == "user":
                if m.tool_results:
                    out.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.tool_call_id,
                                "content": r.content,
                            }
                            for r in m.tool_results
                        ],
                    })
                else:
                    out.append({"role": "user", "content": m.text or ""})
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.text:
                    blocks.append({"type": "text", "text": m.text})
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                out.append({"role": "assistant", "content": blocks})
        return out

    def chat(self, messages: list[Message], tools: list[dict], system: str) -> LLMResponse:
        kwargs = dict(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=self._to_anthropic(messages),
        )
        if tools:
            kwargs["tools"] = tools

        resp = self.client.messages.create(**kwargs)

        text = None
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return LLMResponse(text=text, tool_calls=tool_calls, stop_reason=resp.stop_reason)


# ─────────────────────────────────────────────
# OpenAI / DeepSeek Adapter（兼容 OpenAI 格式）
# ─────────────────────────────────────────────

class OpenAIAdapter(LLMAdapter):
    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str = None,
        api_key: str = None,
        timeout: float | None = None,
        max_retries: int = 0,
    ):
        import httpx
        from openai import OpenAI
        request_timeout = httpx.Timeout(
            connect=10.0,
            read=timeout or 60.0,
            write=10.0,
            pool=10.0,
        )
        self.timeout = request_timeout
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=max_retries,
        )
        self.model = model

    @staticmethod
    def _to_openai(messages: list[Message]) -> list[dict]:
        import json
        # OpenAI 兼容接口要求工具结果用 role=tool 单独回传；
        # 这里把内部统一 Message 历史翻译成 API 所需格式。
        out: list[dict] = []
        for m in messages:
            if m.role == "user":
                if m.tool_results:
                    # OpenAI 要求每个 tool_result 单独一条 role=tool 消息
                    for r in m.tool_results:
                        out.append({
                            "role": "tool",
                            "tool_call_id": r.tool_call_id,
                            "content": r.content,
                        })
                else:
                    out.append({"role": "user", "content": m.text or ""})
            elif m.role == "assistant":
                msg: dict = {"role": "assistant", "content": m.text or None}
                if m.reasoning_content:
                    msg["reasoning_content"] = m.reasoning_content
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.input or {}, ensure_ascii=False),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
        return out

    def chat(self, messages: list[Message], tools: list[dict], system: str) -> LLMResponse:
        import json
        from metrics import console_timer

        full_messages = [{"role": "system", "content": system}] + self._to_openai(messages)

        oai_tools = [
            # 项目自己的 tool schema 转成 OpenAI function calling schema。
            {"type": "function", "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }}
            for t in tools
        ] if tools else None

        with console_timer("LLM API", f"model={self.model} messages={len(full_messages)} tools={len(tools or [])}"):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                tools=oai_tools,
                timeout=self.timeout,
            )

        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))

        # DeepSeek V4 推理链
        # reasoning_content 保存进历史，保证多步工具调用时模型能延续自己的推理状态。
        reasoning = getattr(msg, "reasoning_content", None)

        return LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            reasoning_content=reasoning,
        )


# 切换示例：
# llm = ClaudeAdapter(model="claude-opus-4-5")
# llm = OpenAIAdapter(model="gpt-4o")
# llm = OpenAIAdapter(model="deepseek-chat", base_url="https://api.deepseek.com/v1", api_key="sk-...")
