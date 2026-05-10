"""
agent.py -- Agent 核心循环
"""

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from adapters import LLMAdapter, LLMResponse, Message, ToolResult
from tools import ToolExecutor, READ_TOOLS, WRITE_TOOLS, WRITE_TOOL_NAMES
from memory import MemoryManager
from metrics import console_timer, get_tracer


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


WRITE_INTENT_TERMS = (
    "保存", "存下来", "记录", "记一下", "保存决策",
    "加入持仓", "更新持仓", "记录买入", "清仓", "删除持仓", "移除持仓",
    "加入自选", "加到自选", "关注这只", "加到观察列表", "移出自选", "删除自选",
    "保存复盘", "记录复盘",
)

HISTORY_CONTEXT_TERMS = (
    "上次", "以前", "之前", "历史", "复盘", "决策", "记录",
    "为什么关注", "当时", "持仓", "自选",
)

REALTIME_QUERY_TERMS = (
    "行情", "现价", "当前", "今天", "估值", "pe", "pb",
    "股息率", "分红", "比较", "对比",
)

STOCK_CONTEXT_TERMS = (
    "股票", "a股", "个股", "代码", "估值", "行情", "股息率", "分红", "pe", "pb",
)


class Agent:
    def __init__(self, llm: LLMAdapter, max_steps: int = 20):
        self.llm = llm
        self.memory = MemoryManager()
        self.executor = ToolExecutor(self.memory)
        self.max_steps = max_steps
        self.history: list[Message] = []
        self.pending_write_calls = []

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

    @staticmethod
    def _has_stock_code(text: str) -> bool:
        return bool(re.search(r"(?<!\d)\d{6}(?!\d)", text or ""))

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        text = (text or "").lower()
        return any(term.lower() in text for term in terms)

    @classmethod
    def _has_write_intent(cls, user_query: str) -> bool:
        return cls._contains_any(user_query, WRITE_INTENT_TERMS)

    @classmethod
    def _should_retrieve_memory(cls, user_query: str) -> bool:
        policy = os.environ.get("MEMORY_RETRIEVE_POLICY", "conservative_skip").lower()
        if policy == "always":
            return True

        if cls._contains_any(user_query, HISTORY_CONTEXT_TERMS):
            return True

        has_realtime_term = cls._contains_any(user_query, REALTIME_QUERY_TERMS)
        has_stock_context = (
            cls._has_stock_code(user_query)
            or cls._contains_any(user_query, STOCK_CONTEXT_TERMS)
        )
        if policy in ("conservative_skip", "skip_realtime") and has_realtime_term and has_stock_context:
            return False

        return True

    @classmethod
    def _select_tools_for_turn(cls, user_query: str) -> list[dict]:
        if cls._has_write_intent(user_query):
            return READ_TOOLS + WRITE_TOOLS
        return READ_TOOLS

    def _build_system_prompt(self, user_query: str) -> str:
        portfolio_context = self._build_portfolio_context()
        if self._should_retrieve_memory(user_query):
            memory_context = self.memory.retrieve_context(user_query)
        else:
            memory_context = "（本轮为实时数据查询，已跳过历史记忆检索）"
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
            reasoning_content=response.reasoning_content,
        ))

    def _append_tool_results(self, tool_calls, results: list[str]):
        self.history.append(Message(
            role="user",
            tool_results=[
                ToolResult(tool_call_id=tc.id, content=result)
                for tc, result in zip(tool_calls, results)
            ],
        ))

    # ── 写工具确认 ───────────────────────────

    @staticmethod
    def _is_confirm(text: str) -> bool:
        return text.strip().lower() in {"确认", "确认执行", "是", "yes", "y"}

    @staticmethod
    def _is_cancel(text: str) -> bool:
        return text.strip().lower() in {"取消", "放弃", "不用了", "no", "n"}

    @staticmethod
    def _write_tool_calls(tool_calls) -> list:
        return [tc for tc in tool_calls if tc.name in WRITE_TOOL_NAMES]

    def _format_pending_write_confirmation(self, tool_calls) -> str:
        lines = ["检测到写入操作，尚未执行。请确认是否执行："]
        for tc in tool_calls:
            payload = json.dumps(tc.input or {}, ensure_ascii=False, indent=2)
            lines.append(f"\n工具：{tc.name}\n参数：\n{payload}")
        lines.append("\n回复“确认”执行，回复“取消”放弃。")
        return "\n".join(lines)

    def _execute_pending_write_calls(self) -> str:
        results = []
        for tc in self.pending_write_calls:
            print(f"  [确认写工具] {tc.name}  参数={tc.input}")
            result = self.executor.execute(tc.name, tc.input, allow_write=True)
            results.append(f"- {tc.name}: {result}")
        self.pending_write_calls = []
        return "已执行确认的写操作：\n" + "\n".join(results)

    def _handle_pending_write(self, user_input: str) -> str | None:
        if not self.pending_write_calls:
            return None

        if self._is_confirm(user_input):
            return self._execute_pending_write_calls()

        if self._is_cancel(user_input):
            self.pending_write_calls = []
            return "已取消待确认的写操作。"

        # 用户转向新话题时，旧的待确认写操作失效，避免陈旧写入。
        self.pending_write_calls = []
        return None

    # ── 工具执行 ─────────────────────────────

    def _execute_read_tool_call(self, tc) -> str:
        print(f"  [工具调用] {tc.name}  参数={tc.input}")
        with console_timer("对话阶段", f"tool {tc.name}"):
            return self.executor.execute(tc.name, tc.input)

    def _execute_read_tool_calls(self, tool_calls) -> list[str]:
        if len(tool_calls) <= 1:
            return [self._execute_read_tool_call(tc) for tc in tool_calls]

        max_workers = max(1, int(os.environ.get("READ_TOOL_MAX_WORKERS", "5")))
        max_workers = min(max_workers, len(tool_calls))
        with console_timer("对话阶段", f"parallel read tools n={len(tool_calls)} workers={max_workers}"):
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(self._execute_read_tool_call, tool_calls))

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
                get_tracer().bump_memory_writes()
        except Exception as e:
            print(f"  [记忆写入失败] {e}")

    def _save_conversation_insight_after_response(self, user_input: str, final_response: str):
        async_enabled = os.environ.get("ASYNC_MEMORY_WRITE", "true").lower() not in (
            "0", "false", "no", "off"
        )
        if not async_enabled:
            with console_timer("对话阶段", "对话摘要写入"):
                self._save_conversation_insight(user_input, final_response)
            return

        def run():
            with console_timer("对话阶段", "对话摘要写入 async"):
                self._save_conversation_insight(user_input, final_response)

        threading.Thread(target=run, daemon=True).start()

    # ── 主循环 ───────────────────────────────

    def chat(self, user_input: str) -> str:
        tracer = get_tracer()

        with tracer.turn(user_input, self.history):
            pending_result = self._handle_pending_write(user_input)
            if pending_result is not None:
                self._append_user(user_input)
                self._append_assistant(LLMResponse(text=pending_result))
                tracer.set_history_end(self.history, pending_result)
                self._save_conversation_insight_after_response(user_input, pending_result)
                return pending_result

            self._append_user(user_input)
            with console_timer("对话阶段", "构建system prompt"):
                system = self._build_system_prompt(user_input)
            tracer.set_system_prompt(system)
            active_tools = self._select_tools_for_turn(user_input)

            steps = 0
            final_text = ""

            while steps < self.max_steps:
                steps += 1
                with tracer.llm_call(system, self.history) as llm_rec:
                    with console_timer("对话阶段", f"LLM step={steps}"):
                        response = self.llm.chat(self.history, active_tools, system)
                    llm_rec.output_text_chars = len(response.text or "")
                    llm_rec.tool_calls_emitted = len(response.tool_calls)

                write_calls = self._write_tool_calls(response.tool_calls)
                if write_calls:
                    self.pending_write_calls = write_calls
                    final_text = self._format_pending_write_confirmation(write_calls)
                    self._append_assistant(LLMResponse(text=final_text))
                    break

                self._append_assistant(response)

                if not response.tool_calls:
                    final_text = response.text or ""
                    break

                results = self._execute_read_tool_calls(response.tool_calls)

                self._append_tool_results(response.tool_calls, results)

            else:
                final_text = f"[警告] 达到最大步数 {self.max_steps}，强制终止。"

            tracer.set_history_end(self.history, final_text)
            self._save_conversation_insight_after_response(user_input, final_text)

        return final_text

    def reset(self):
        self.history = []
        self.pending_write_calls = []
        print("对话已清空（记忆和数据库保留）")
