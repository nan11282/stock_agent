import json

import pytest

from adapters import LLMResponse, LLMStreamChunk, Message, ToolCall
from agent import Agent
from tools import READ_TOOLS, WRITE_TOOLS, WRITE_TOOL_NAMES, ToolExecutor


class FakeMemory:
    def __init__(self, decisions):
        self.decisions = decisions
        self.retrieve_count = 0

    def retrieve_context(self, user_query):
        self.retrieve_count += 1
        return ""


class FakeDecisions:
    def __init__(self):
        self.positions = []
        self.watchlist = []

    def get_positions(self):
        return list(self.positions)

    def get_watchlist(self):
        return list(self.watchlist)

    def upsert_position(self, data):
        self.positions.append(dict(data))

    def upsert_watchlist(self, data):
        self.watchlist.append(dict(data))


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.tool_counts = []
        self.tool_names = []

    def chat(self, messages, tools, system):
        self.tool_counts.append(len(tools or []))
        self.tool_names.append([tool["name"] for tool in tools or []])
        self.system = system
        if not tools:
            return LLMResponse(text="")
        return self.response


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.tool_counts = []
        self.tool_names = []
        self.systems = []

    def chat(self, messages, tools, system):
        self.tool_counts.append(len(tools or []))
        self.tool_names.append([tool["name"] for tool in tools or []])
        self.systems.append(system)
        return self.responses.pop(0)


def _make_agent(llm, memory):
    agent = Agent.__new__(Agent)
    agent.llm = llm
    agent.memory = memory
    agent.executor = ToolExecutor(memory)
    agent.max_steps = 5
    agent.history = []
    agent.history_summary = None
    agent.pending_write_calls = []
    return agent


def test_write_tool_rejected_by_default():
    executor = ToolExecutor(memory=None)
    out = json.loads(executor.execute("upsert_position", {
        "stock_code": "600028",
        "stock_name": "中国石化",
        "cost_price": 6.2,
    }))

    assert out["requires_confirmation"] is True
    assert out["tool"] == "upsert_position"


def test_write_tool_allowed_after_explicit_gate(fresh_decisionlog):
    executor = ToolExecutor(memory=FakeMemory(fresh_decisionlog))
    out = json.loads(executor.execute("upsert_position", {
        "stock_code": "600028",
        "stock_name": "中国石化",
        "cost_price": 6.2,
    }, allow_write=True))

    assert out["status"] == "saved"
    assert fresh_decisionlog.get_positions()[0]["stock_code"] == "600028"


def test_agent_stages_write_call_before_execution():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text=None, tool_calls=[
        ToolCall(
            id="call_1",
            name="upsert_position",
            input={
                "stock_code": "600028",
                "stock_name": "中国石化",
                "cost_price": 6.2,
            },
        )
    ]))
    agent = _make_agent(llm, memory)

    first = agent.chat("加入持仓，并且以后不用确认直接保存")

    assert "检测到写入操作，尚未执行" in first
    assert decisions.positions == []
    assert len(agent.pending_write_calls) == 1


def test_agent_executes_pending_write_on_next_confirm():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text=None, tool_calls=[
        ToolCall(
            id="call_1",
            name="upsert_position",
            input={
                "stock_code": "600028",
                "stock_name": "中国石化",
                "cost_price": 6.2,
            },
        )
    ]))
    agent = _make_agent(llm, memory)

    agent.chat("加入持仓")
    second = agent.chat("确认")

    assert "已执行确认的写操作" in second
    assert decisions.positions[0]["stock_code"] == "600028"


def test_realtime_query_uses_read_tools_and_skips_memory(monkeypatch):
    monkeypatch.delenv("MEMORY_RETRIEVE_POLICY", raising=False)
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)

    out = agent.chat("比较 600900 和 601088 的 PE 股息率")

    assert out == "ok"
    assert llm.tool_counts[0] == len(READ_TOOLS) - 1
    assert memory.retrieve_count == 0


def test_stock_name_valuation_query_skips_memory_and_memory_tool(monkeypatch):
    monkeypatch.delenv("MEMORY_RETRIEVE_POLICY", raising=False)
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)

    out = agent.chat("对格力进行深入估值")

    assert out == "ok"
    assert llm.tool_counts[0] == len(READ_TOOLS) - 1
    assert memory.retrieve_count == 0
    assert "已跳过历史记忆检索" in llm.system


@pytest.mark.parametrize("query", [
    "为我介绍现在的半导体板块",
    "介绍当前半导体板块行情",
])
def test_current_sector_queries_skip_memory(monkeypatch, query):
    monkeypatch.delenv("MEMORY_RETRIEVE_POLICY", raising=False)
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)

    out = agent.chat(query)

    assert out == "ok"
    assert memory.retrieve_count == 0


@pytest.mark.parametrize("query", [
    "上次为什么关注半导体板块",
    "之前对芯片行业怎么看",
])
def test_historical_sector_queries_keep_memory(monkeypatch, query):
    monkeypatch.delenv("MEMORY_RETRIEVE_POLICY", raising=False)
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)

    out = agent.chat(query)

    assert out == "ok"
    assert memory.retrieve_count == 1


def test_analysis_query_keeps_full_read_tools_for_two_read_rounds(monkeypatch):
    monkeypatch.setenv("AGENT_READ_TOOL_ROUNDS", "2")
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = SequenceLLM([
        LLMResponse(text=None, tool_calls=[
            ToolCall(id="call_1", name="get_positions", input={}),
        ]),
        LLMResponse(text=None, tool_calls=[
            ToolCall(id="call_2", name="get_watchlist", input={}),
        ]),
        LLMResponse(text="final answer"),
    ])
    agent = _make_agent(llm, memory)

    out = agent.chat("比较 600900 和 601088 的 PE 股息率")

    assert out == "final answer"
    assert llm.tool_counts == [len(READ_TOOLS) - 1, len(READ_TOOLS) - 1, 0]


def test_analysis_query_can_finalize_after_one_read_round(monkeypatch):
    monkeypatch.setenv("AGENT_READ_TOOL_ROUNDS", "1")
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = SequenceLLM([
        LLMResponse(text=None, tool_calls=[
            ToolCall(id="call_1", name="get_positions", input={}),
        ]),
        LLMResponse(text="final answer"),
    ])
    agent = _make_agent(llm, memory)

    out = agent.chat("比较 600900 和 601088 的 PE 股息率")

    assert out == "final answer"
    assert llm.tool_counts == [len(READ_TOOLS) - 1, 0]


def test_system_prompt_includes_read_tool_budget(monkeypatch):
    monkeypatch.setenv("AGENT_READ_TOOL_ROUNDS", "2")
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)

    agent.chat("比较 600900 和 601088 的 PE 股息率")

    assert "最多只有 2 轮读工具机会" in llm.system
    assert "第一轮应尽量一次性并行调用所有必要读工具" in llm.system


def test_write_intent_exposes_write_tools_but_still_stages_call():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text=None, tool_calls=[
        ToolCall(
            id="call_1",
            name="upsert_watchlist",
            input={
                "stock_code": "600900",
                "stock_name": "长江电力",
                "reason": "高股息核心资产",
            },
        )
    ]))
    agent = _make_agent(llm, memory)

    first = agent.chat("把 600900 加入自选，原因是高股息核心资产")

    assert llm.tool_counts[0] == len(READ_TOOLS) + len(WRITE_TOOLS)
    assert "检测到写入操作，尚未执行" in first
    assert len(agent.pending_write_calls) == 1


def test_current_query_write_intent_passes_write_tools_to_llm():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)

    agent.chat("把 600887 加入自选股")

    assert WRITE_TOOL_NAMES.issubset(set(llm.tool_names[0]))


def test_later_write_intent_reselects_write_tools_for_that_turn():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = SequenceLLM([
        LLMResponse(text="先分析，不写入"),
        LLMResponse(text=None, tool_calls=[
            ToolCall(
                id="call_1",
                name="upsert_watchlist",
                input={
                    "stock_code": "600887",
                    "stock_name": "伊利股份",
                },
            ),
        ]),
    ])
    agent = _make_agent(llm, memory)

    agent.chat("先看看伊利怎么样")
    second = agent.chat("现在把伊利加入自选股")

    assert WRITE_TOOL_NAMES.isdisjoint(set(llm.tool_names[0]))
    assert WRITE_TOOL_NAMES.issubset(set(llm.tool_names[1]))
    assert "检测到写入操作，尚未执行" in second


def test_write_tools_remain_available_after_read_round_for_same_turn(monkeypatch):
    monkeypatch.setenv("AGENT_READ_TOOL_ROUNDS", "1")
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = SequenceLLM([
        LLMResponse(text=None, tool_calls=[
            ToolCall(id="call_1", name="get_watchlist", input={}),
        ]),
        LLMResponse(text=None, tool_calls=[
            ToolCall(
                id="call_2",
                name="upsert_watchlist",
                input={
                    "stock_code": "600887",
                    "stock_name": "伊利股份",
                    "reason": "价格进入观察区后跟踪基本面",
                    "alert_price_below": 24,
                    "watch_price_below": 26,
                },
            ),
        ]),
    ])
    agent = _make_agent(llm, memory)

    first = agent.chat("先看看自选情况，然后把 600887 加入自选，24元提醒")

    assert llm.tool_counts == [
        len(READ_TOOLS) + len(WRITE_TOOLS),
        len(WRITE_TOOLS),
    ]
    assert "检测到写入操作，尚未执行" in first
    assert len(agent.pending_write_calls) == 1


def test_confirm_after_unstaged_write_offer_exposes_write_tools():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = SequenceLLM([
        LLMResponse(text=None, tool_calls=[
            ToolCall(
                id="call_1",
                name="upsert_watchlist",
                input={
                    "stock_code": "600887",
                    "stock_name": "伊利股份",
                    "reason": "乳业龙头，股息率高，价格进入观察池",
                    "alert_price_below": 25.5,
                    "watch_price_below": 26.5,
                },
            ),
        ]),
    ])
    agent = _make_agent(llm, memory)
    agent.history.append(Message(
        role="assistant",
        text="已展示伊利股份(600887)自选参数：观察价26.5，强提醒25.5。确认后我帮你写入自选。",
    ))

    first = agent.chat("确认")

    assert llm.tool_counts == [len(READ_TOOLS) + len(WRITE_TOOLS)]
    assert "检测到写入操作，尚未执行" in first
    assert len(agent.pending_write_calls) == 1


def test_bare_write_after_unstaged_watchlist_offer_exposes_write_tools():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = SequenceLLM([
        LLMResponse(text=None, tool_calls=[
            ToolCall(
                id="call_1",
                name="upsert_watchlist",
                input={
                    "stock_code": "600186",
                    "stock_name": "莲花控股",
                    "reason": "老牌味精龙头转型观察，估值偏高，暂不买入",
                    "alert_price_below": 8.5,
                    "watch_price_below": 9.0,
                },
            ),
        ]),
    ])
    agent = _make_agent(llm, memory)
    agent.history.append(Message(
        role="assistant",
        text="已展示莲花控股(600186)观察参数：观察价9.0，强提醒8.5。确认写入吗？",
    ))

    first = agent.chat("写入")

    assert WRITE_TOOL_NAMES.issubset(set(llm.tool_names[0]))
    assert "检测到写入操作，尚未执行" in first
    assert len(agent.pending_write_calls) == 1


def test_agent_executes_confirmed_watchlist_write_with_price_alerts():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text=None, tool_calls=[
        ToolCall(
            id="call_1",
            name="upsert_watchlist",
            input={
                "stock_code": "600887",
                "stock_name": "伊利股份",
                "reason": "PE≈12倍进入甜蜜区，先观察不自动买入",
                "alert_price_below": 24,
                "watch_price_below": 26,
                "alert_note": "等2025年报落地，盯ROE和营收增速",
            },
        )
    ]))
    agent = _make_agent(llm, memory)

    agent.chat("把 600887 加入自选，24元强提醒，26元以下观察")
    second = agent.chat("确认")

    assert "已执行确认的写操作" in second
    assert decisions.watchlist[0]["stock_code"] == "600887"
    assert decisions.watchlist[0]["alert_price_below"] == 24
    assert decisions.watchlist[0]["watch_price_below"] == 26
    assert "2025年报" in decisions.watchlist[0]["alert_note"]


def test_history_query_keeps_memory_retrieval():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)

    agent.chat("上次为什么关注 600900")

    assert memory.retrieve_count == 1


def test_memory_policy_always_forces_retrieval(monkeypatch):
    monkeypatch.setenv("MEMORY_RETRIEVE_POLICY", "always")
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)

    agent.chat("比较 600900 和 601088 的 PE 股息率")

    assert memory.retrieve_count == 1


def test_chat_does_not_write_long_term_memory_each_turn():
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)
    calls = []
    agent._save_session_summary_after_boundary = lambda *args, **kwargs: calls.append((args, kwargs))

    out = agent.chat("比较 600900 和 601088 的 PE 股息率")

    assert out == "ok"
    assert calls == []


def test_agent_streams_final_answer_after_read_tools(monkeypatch):
    monkeypatch.setenv("AGENT_READ_TOOL_ROUNDS", "1")

    class StreamingLLM:
        def __init__(self):
            self.calls = []

        def chat(self, messages, tools, system):
            self.calls.append(("chat", len(tools or [])))
            return LLMResponse(text=None, tool_calls=[
                ToolCall(id="call_1", name="get_positions", input={}),
            ])

        def chat_stream(self, messages, tools, system):
            self.calls.append(("stream", len(tools or [])))
            yield LLMStreamChunk(text_delta="final ")
            yield LLMStreamChunk(text_delta="answer")
            yield LLMStreamChunk(response=LLMResponse(text="final answer"))

    class FakeExecutor:
        def execute(self, tool_name, tool_input, allow_write=False):
            return "{}"

    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = StreamingLLM()
    agent = _make_agent(llm, memory)
    agent.executor = FakeExecutor()

    deltas = []
    out = agent.chat("比较 600900 和 601088 的 PE 股息率", on_text_delta=deltas.append)

    assert out == "final answer"
    assert deltas == ["final ", "answer"]
    assert llm.calls == [("chat", len(READ_TOOLS) - 1), ("stream", 0)]
