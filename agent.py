"""
agent.py -- Agent 核心循环
"""

from adapters import LLMAdapter, LLMResponse, Message, ToolResult
from tools import ToolExecutor, ALL_TOOLS
from memory import MemoryManager


SYSTEM_PROMPT_TEMPLATE = """
你是 Coco 的私人A股投资助理。

【投资哲学】
- 核心仓：高股息、低估值的大盘蓝筹（银行、能源、公用事业）
- 成长仓：有护城河、ROE稳定的优质公司
- 估值框架：TTM股息率、PE历史百分位、AH溢价

【当前持仓】
{portfolio_context}

【相关历史记忆】
{memory_context}

【工具使用规则——不得违反】
读工具：分析过程中自主调用，无需请示。
写工具：必须同时满足：
  1. 用户明确说了操作指令
  2. 已向用户展示内容并收到"确认"
分析讨论过程中绝对禁止写入数据库。

【风格】
数据驱动，先调工具拿真实数据再分析。
给明确观点，不模糊。复盘时直接指出错误。
"""


class Agent:
    def __init__(self, llm: LLMAdapter, max_steps: int = 20):
        self.llm = llm
        self.memory = MemoryManager()
        self.executor = ToolExecutor(self.memory)
        self.max_steps = max_steps
        self.history: list[Message] = []

    # ── Prompt 构建 ──────────────────────────

    def _build_portfolio_context(self) -> str:
        parts = []

        positions = self.memory.decisions.get_positions()
        if positions:
            parts.append("持仓：")
            for p in positions:
                line = f"  {p['stock_name']}({p['stock_code']}) 成本{p['cost_price']}"
                if p.get("position_pct"):
                    line += f" 仓位{p['position_pct']}%"
                if p.get("tier"):
                    line += f" [{p['tier']}]"
                parts.append(line)
        else:
            parts.append("持仓：暂无")

        watchlist = self.memory.decisions.get_watchlist()
        if watchlist:
            parts.append("自选股：")
            for w in watchlist:
                line = f"  {w['stock_name']}({w['stock_code']})"
                if w.get("reason"):
                    line += f" — {w['reason']}"
                alerts = []
                if w.get("alert_yield"):
                    alerts.append(f"股息率>{w['alert_yield']}%")
                if w.get("alert_pe_pct"):
                    alerts.append(f"PE百分位<{w['alert_pe_pct']}")
                if alerts:
                    line += f" 提醒: {', '.join(alerts)}"
                parts.append(line)

        return "\n".join(parts) if parts else "暂无持仓和自选股"

    def _build_system_prompt(self, user_query: str) -> str:
        portfolio_context = self._build_portfolio_context()
        memory_context = self.memory.retrieve_context(user_query)
        return SYSTEM_PROMPT_TEMPLATE.format(
            portfolio_context=portfolio_context,
            memory_context=memory_context or "（暂无相关历史记忆）",
        )

    # ── History 管理 ─────────────────────────

    def _append_user(self, text: str):
        self.history.append(Message(role="user", text=text))

    def _append_assistant(self, response: LLMResponse):
        self.history.append(Message(
            role="assistant",
            text=response.text,
            tool_calls=list(response.tool_calls),
        ))

    def _append_tool_results(self, tool_calls, results: list[str]):
        self.history.append(Message(
            role="user",
            tool_results=[
                ToolResult(tool_call_id=tc.id, content=result)
                for tc, result in zip(tool_calls, results)
            ],
        ))

    # ── 对话后提炼摘要写入向量库+FTS5 ────────

    def _save_conversation_insight(self, user_input: str, final_response: str):
        summary_prompt = [
            Message(role="user", text=(
                f"请将以下这段投资对话总结成2-3句话，"
                f"包含：讨论的股票代码和名称、核心观点、结论或待观察点。\n\n"
                f"用户问：{user_input}\n"
                f"助理答：{final_response[:500]}"
            ))
        ]
        try:
            resp = self.llm.chat(
                messages=summary_prompt,
                tools=[],
                system="你是一个投资记录助手，只输出简洁的摘要，不超过100字。",
            )
            if resp.text:
                self.memory.episodic.save_insight(
                    text=resp.text,
                    metadata={"source_query": user_input[:100]},
                )
        except Exception as e:
            print(f"  [记忆写入失败] {e}")

    # ── 主循环 ───────────────────────────────

    def chat(self, user_input: str) -> str:
        self._append_user(user_input)
        system = self._build_system_prompt(user_input)

        steps = 0
        final_text = ""

        while steps < self.max_steps:
            steps += 1
            response = self.llm.chat(self.history, ALL_TOOLS, system)
            self._append_assistant(response)

            if not response.tool_calls:
                final_text = response.text or ""
                break

            results = []
            for tc in response.tool_calls:
                print(f"  [工具调用] {tc.name}  参数={tc.input}")
                result = self.executor.execute(tc.name, tc.input)
                results.append(result)

            self._append_tool_results(response.tool_calls, results)

        else:
            final_text = f"[警告] 达到最大步数 {self.max_steps}，强制终止。"

        self._save_conversation_insight(user_input, final_text)

        return final_text

    def reset(self):
        self.history = []
        print("对话已清空（记忆和数据库保留）")
