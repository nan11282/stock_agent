"""
agent.py -- Agent 核心循环
"""

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from adapters import LLMAdapter, LLMResponse, Message, ToolResult
from tools import ToolExecutor, READ_TOOLS, WRITE_TOOLS, WRITE_TOOL_NAMES, resolve_stock_code
from memory import MemoryManager
from metrics import console_timer, get_tracer, history_chars


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
读工具预算：本轮最多只有 {read_tool_rounds} 轮读工具机会。
第一轮应尽量一次性并行调用所有必要读工具。
第二轮只用于补漏，不要重复读取已经拿到的数据。
工具机会用完后，必须基于已有数据回答；若关键数据缺失，要明确说明缺失项和影响。
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

# 这些词表是 Agent 的“业务路由器”：先用可解释规则判断用户是在复盘、
# 查实时行情，还是准备写入记录，再决定是否加载历史记忆、是否暴露写工具。
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

CHINESE_NUMBERS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_small_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in CHINESE_NUMBERS:
        return CHINESE_NUMBERS[text]
    if "十" in text:
        left, _, right = text.partition("十")
        tens = CHINESE_NUMBERS.get(left, 1) if left else 1
        ones = CHINESE_NUMBERS.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


class Agent:
    def __init__(self, llm: LLMAdapter, max_steps: int = 20):
        self.llm = llm
        self.memory = MemoryManager()
        self.executor = ToolExecutor(self.memory)
        self.max_steps = max_steps
        self.history: list[Message] = []
        self.history_summary: str | None = None
        self.pending_write_calls = []

    # ── Prompt 构建 ──────────────────────────

    def _build_portfolio_context(self) -> str:
        # 每轮对话都把当前持仓/自选塞进 system prompt。
        # 这样模型在给建议时天然知道用户已有风险暴露，而不是孤立分析一只股票。
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

        # 复盘、历史决策、持仓原因这类问题必须查记忆；
        # 纯实时行情问题默认跳过，避免旧观点污染最新数据判断。
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
        # 写工具只在用户有明确保存/更新/删除意图时才暴露给 LLM。
        # 后续仍需二次确认，这里只是第一道业务权限门。
        if cls._has_write_intent(user_query):
            return READ_TOOLS + WRITE_TOOLS
        return READ_TOOLS

    @staticmethod
    def _read_tool_round_limit() -> int:
        return max(0, _env_int("AGENT_READ_TOOL_ROUNDS", 2))

    @classmethod
    def _is_dividend_reinvestment_query(cls, user_query: str) -> bool:
        text = (user_query or "").lower()
        # 股息复投是高频、结构化的理财测算场景，直接走确定性计算，
        # 不让 LLM 在多轮工具调用里自由发挥，减少口径漂移。
        return (
            "定投" in text
            and "股息" in text
            and "复投" in text
            and any(term in text for term in ("年", "收益", "当年", "总收益"))
        )

    @staticmethod
    def _parse_monthly_lots(user_query: str) -> int | None:
        match = re.search(r"每个?月[^，。,.]*?([0-9]+|[一二两三四五六七八九十]+)\s*手", user_query or "")
        if not match:
            match = re.search(r"([0-9]+|[一二两三四五六七八九十]+)\s*手", user_query or "")
        return _parse_small_int(match.group(1)) if match else None

    @staticmethod
    def _parse_years(user_query: str) -> int | None:
        match = re.search(r"([0-9]+|[一二两三四五六七八九十]+)\s*年", user_query or "")
        return _parse_small_int(match.group(1)) if match else None

    @staticmethod
    def _format_dividend_reinvestment_result(result: dict) -> str:
        if result.get("error"):
            return f"[计算失败] {result['error']}"

        yearly_tail = result.get("yearly", [])[-3:]
        tail_lines = []
        for item in yearly_tail:
            tail_lines.append(
                f"- 第{item['year']}年：当年股息 {item['cash_dividend']:,.2f} 元，"
                f"复投 {item['reinvested_shares']:,} 股，年末持股 {item['ending_shares']:,} 股"
            )

        return "\n".join([
            f"按当前静态假设测算：{result['stock_name']}({result['stock_code']})",
            "",
            f"- 当前价格：{result['price']} 元",
            f"- 采用每股年分红：{result['annual_dividend_per_share']} 元"
            + (f"（来源日期 {result['dividend_source_date']}）" if result.get("dividend_source_date") else ""),
            f"- 每月定投：{result['monthly_lots']} 手 = {result['monthly_shares']:,} 股",
            f"- 测算年限：{result['years']} 年，股息复投：{'是' if result['dividend_reinvest'] else '否'}",
            "",
            f"结果：",
            f"- 定投买入股数：{result['regular_buy_shares']:,} 股",
            f"- 股息复投新增股数：{result['reinvested_shares']:,} 股",
            f"- {result['years']}年末总持股：{result['ending_shares']:,} 股",
            f"- {result['years']}年累计股息：{result['cumulative_dividend']:,.2f} 元",
            f"- 第{result['years']}年当年股息：{result['last_year_dividend']:,.2f} 元",
            f"- 复投后剩余现金：{result['remaining_cash']:,.2f} 元",
            "",
            "最近3年明细：",
            *tail_lines,
            "",
            f"假设：{result['assumption']}这是静态估算，不构成投资建议。",
        ])

    def _handle_dividend_reinvestment_query(self, user_input: str) -> str | None:
        if not self._is_dividend_reinvestment_query(user_input):
            return None

        stock_code = resolve_stock_code(user_input)
        monthly_lots = self._parse_monthly_lots(user_input)
        years = self._parse_years(user_input)
        if not stock_code or not monthly_lots or not years:
            return None

        result = json.loads(self.executor.execute("calculate_dividend_reinvestment", {
            "stock_code": stock_code,
            "monthly_lots": monthly_lots,
            "years": years,
            "dividend_reinvest": True,
        }))
        return self._format_dividend_reinvestment_result(result)

    def _build_system_prompt(self, user_query: str) -> str:
        portfolio_context = self._build_portfolio_context()
        # system prompt 是每轮投资判断的“现场材料包”：
        # 当前账户事实 + 必要历史记忆 + 工具预算约束。
        if self._should_retrieve_memory(user_query):
            memory_context = self.memory.retrieve_context(user_query)
        else:
            memory_context = "（本轮为实时数据查询，已跳过历史记忆检索）"
        return SYSTEM_PROMPT_TEMPLATE.format(
            portfolio_context=portfolio_context,
            memory_context=memory_context or "（暂无相关历史记忆）",
            read_tool_rounds=self._read_tool_round_limit(),
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

    @staticmethod
    def _short_memory_max_chars() -> int:
        return max(0, _env_int("SHORT_MEMORY_MAX_CHARS", 24000))

    @staticmethod
    def _short_memory_keep_turns() -> int:
        return max(0, _env_int("SHORT_MEMORY_KEEP_TURNS", 4))

    @staticmethod
    def _short_memory_summary_max_chars() -> int:
        return max(200, _env_int("SHORT_MEMORY_SUMMARY_MAX_CHARS", 3000))

    def _history_summary_text(self) -> str:
        return getattr(self, "history_summary", None) or ""

    def _history_for_llm(self) -> list[Message]:
        summary = self._history_summary_text().strip()
        if not summary:
            return self.history
        # 短期摘要作为一条用户侧上下文注入，保留前情但不把完整流水账继续塞给模型。
        return [
            Message(role="user", text=f"【此前对话压缩摘要】\n{summary}"),
            *self.history,
        ]

    @staticmethod
    def _message_for_compaction(m: Message) -> str:
        if m.role == "user" and m.text:
            return f"用户：{m.text}"

        if m.role == "assistant":
            parts = []
            if m.text:
                parts.append(f"助理：{m.text}")
            for tc in m.tool_calls:
                payload = json.dumps(tc.input or {}, ensure_ascii=False)
                parts.append(f"助理调用工具：{tc.name} 参数={payload}")
            return "\n".join(parts) if parts else "助理：（空回复）"

        if m.role == "user" and m.tool_results:
            parts = []
            for r in m.tool_results:
                content = r.content or ""
                if len(content) > 1200:
                    content = content[:1200] + "...[已截断]"
                parts.append(f"工具结果({r.tool_call_id})：{content}")
            return "\n".join(parts)

        return f"{m.role}：{m.text or ''}"

    @classmethod
    def _messages_for_compaction(cls, messages: list[Message]) -> str:
        return "\n\n".join(cls._message_for_compaction(m) for m in messages)

    @staticmethod
    def _last_user_turn_start(messages: list[Message], keep_turns: int) -> int | None:
        if keep_turns <= 0:
            return len(messages)

        starts = [
            i for i, m in enumerate(messages)
            if m.role == "user" and m.text
        ]
        if len(starts) <= keep_turns:
            return None
        return starts[-keep_turns]

    def _compact_history(self, old_messages: list[Message]) -> str | None:
        old_summary = self._history_summary_text().strip()
        old_context = old_summary or "（无）"
        transcript = self._messages_for_compaction(old_messages)
        summary_max_chars = self._short_memory_summary_max_chars()
        prompt = (
            # 摘要目标不是“压缩文本”，而是保住未来投资判断仍需要的状态：
            # 用户偏好、已形成观点、待确认事项和关键数据。
            "请把以下A股投资助理的较早对话压缩成稳定的短期会话摘要。\n"
            "目标：后续模型只看摘要和最近原文，也能延续用户偏好、约束、已形成观点与未完成事项。\n\n"
            "必须保留：\n"
            "- 用户投资偏好、风险约束、长期关注方向。\n"
            "- 讨论过的股票代码/名称、关键结论、待验证事项。\n"
            "- 已明确的用户指令、仍需跟进或确认的事项。\n"
            "- 支撑明确投资结论的关键数据。\n\n"
            "不要保留：一次性寒暄、重复过程、无结论的临时行情数字、工具调用流水账。\n"
            f"输出不超过 {summary_max_chars} 字，只输出摘要正文。\n\n"
            f"【已有压缩摘要】\n{old_context}\n\n"
            f"【本次需要并入摘要的较早对话】\n{transcript}"
        )
        resp = self.llm.chat(
            messages=[Message(role="user", text=prompt)],
            tools=[],
            system="你是投资对话短期记忆压缩器。只输出事实准确、可延续上下文的中文摘要。",
        )
        text = (resp.text or "").strip()
        if not text:
            return None
        return text[:summary_max_chars]

    def _maybe_compact_history(self):
        max_chars = self._short_memory_max_chars()
        if max_chars <= 0:
            return

        summary_chars = len(self._history_summary_text())
        if history_chars(self.history) + summary_chars <= max_chars:
            return

        if getattr(self, "pending_write_calls", None):
            # 等待确认的写操作不能被压缩掉，否则用户一句“确认”可能失去上下文。
            return

        keep_start = self._last_user_turn_start(
            self.history,
            self._short_memory_keep_turns(),
        )
        if keep_start is None or keep_start <= 0:
            return

        old_messages = self.history[:keep_start]
        recent_messages = self.history[keep_start:]
        try:
            with console_timer("对话阶段", f"短期记忆压缩 old_messages={len(old_messages)}"):
                new_summary = self._compact_history(old_messages)
        except Exception as e:
            print(f"  [短期记忆压缩失败] {e}")
            return

        if not new_summary:
            return
        self.history_summary = new_summary
        self.history = recent_messages
        self._save_session_summary_after_boundary(new_summary, source="auto_compact")

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
        # 所有数据库写入都先转成可读清单给用户确认；
        # 这保护的是资金决策记录的审计性，而不是单纯防误触。
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
        # 行情、分红、财务等读工具彼此独立，并行能显著缩短 Telegram/CLI 等待时间。
        with console_timer("对话阶段", f"parallel read tools n={len(tool_calls)} workers={max_workers}"):
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(self._execute_read_tool_call, tool_calls))

    # ── 会话边界摘要写入向量库+FTS5 ────────

    def _save_session_summary(self, summary: str, source: str):
        text = (summary or "").strip()
        if not text:
            return
        try:
            self.memory.episodic.save_insight(
                text=text,
                metadata={"source": source},
            )
            get_tracer().bump_memory_writes()
        except Exception as e:
            print(f"  [记忆写入失败] {e}")

    def _save_session_summary_after_boundary(self, summary: str, source: str):
        async_enabled = os.environ.get("ASYNC_MEMORY_WRITE", "true").lower() not in (
            "0", "false", "no", "off"
        )
        if not async_enabled:
            with console_timer("对话阶段", f"会话摘要写入 source={source}"):
                self._save_session_summary(summary, source)
            return

        def run():
            with console_timer("对话阶段", f"会话摘要写入 async source={source}"):
                self._save_session_summary(summary, source)

        threading.Thread(target=run, daemon=True).start()

    def _session_summary_for_reset(self) -> str | None:
        summary = self._history_summary_text().strip()
        if not summary and not self.history:
            return None
        if not self.history:
            return summary
        try:
            new_summary = self._compact_history(self.history)
        except Exception as e:
            print(f"  [reset会话压缩失败] {e}")
            return summary or None
        return new_summary or summary or None

    # ── 主循环 ───────────────────────────────

    def chat(self, user_input: str) -> str:
        tracer = get_tracer()

        with tracer.turn(user_input, self.history):
            pending_result = self._handle_pending_write(user_input)
            if pending_result is not None:
                self._append_user(user_input)
                self._append_assistant(LLMResponse(text=pending_result))
                tracer.set_history_end(self.history, pending_result)
                self._maybe_compact_history()
                return pending_result

            direct_result = self._handle_dividend_reinvestment_query(user_input)
            if direct_result is not None:
                self._append_user(user_input)
                self._append_assistant(LLMResponse(text=direct_result))
                tracer.set_history_end(self.history, direct_result)
                self._maybe_compact_history()
                return direct_result

            self._append_user(user_input)
            with console_timer("对话阶段", "构建system prompt"):
                system = self._build_system_prompt(user_input)
            tracer.set_system_prompt(system)
            active_tools = self._select_tools_for_turn(user_input)
            read_tool_round_limit = self._read_tool_round_limit()
            read_tool_rounds_used = 0

            steps = 0
            final_text = ""

            while steps < self.max_steps:
                steps += 1
                llm_history = self._history_for_llm()
                with tracer.llm_call(system, llm_history) as llm_rec:
                    with console_timer("对话阶段", f"LLM step={steps}"):
                        response = self.llm.chat(llm_history, active_tools, system)
                    llm_rec.output_text_chars = len(response.text or "")
                    llm_rec.tool_calls_emitted = len(response.tool_calls)

                write_calls = self._write_tool_calls(response.tool_calls)
                if write_calls:
                    # LLM 只能提出“想写什么”，真正写库必须等用户下一轮确认。
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
                read_tool_rounds_used += 1
                if read_tool_rounds_used >= read_tool_round_limit:
                    # 读工具预算用完后收回工具，让模型基于已有证据给最终结论。
                    active_tools = []
                else:
                    active_tools = READ_TOOLS

            else:
                final_text = f"[警告] 达到最大步数 {self.max_steps}，强制终止。"

            tracer.set_history_end(self.history, final_text)
            self._maybe_compact_history()

        return final_text

    def reset(self):
        try:
            summary = self._session_summary_for_reset()
            if summary:
                self._save_session_summary_after_boundary(summary, source="manual_reset")
        except Exception as e:
            print(f"  [reset会话摘要失败] {e}")
        self.history = []
        self.history_summary = None
        self.pending_write_calls = []
        print("对话已清空（记忆和数据库保留）")
