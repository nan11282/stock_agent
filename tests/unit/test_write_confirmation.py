import json

from adapters import LLMResponse, ToolCall
from agent import Agent
from tools import READ_TOOLS, WRITE_TOOLS, ToolExecutor


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

    def get_positions(self):
        return list(self.positions)

    def get_watchlist(self):
        return []

    def upsert_position(self, data):
        self.positions.append(dict(data))


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.tool_counts = []

    def chat(self, messages, tools, system):
        self.tool_counts.append(len(tools or []))
        self.system = system
        if not tools:
            return LLMResponse(text="")
        return self.response


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.tool_counts = []
        self.systems = []

    def chat(self, messages, tools, system):
        self.tool_counts.append(len(tools or []))
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
    assert llm.tool_counts[0] == len(READ_TOOLS)
    assert memory.retrieve_count == 0


def test_analysis_query_keeps_full_read_tools_for_two_read_rounds(monkeypatch):
    monkeypatch.delenv("AGENT_READ_TOOL_ROUNDS", raising=False)
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
    assert llm.tool_counts == [len(READ_TOOLS), len(READ_TOOLS), 0]


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
    assert llm.tool_counts == [len(READ_TOOLS), 0]


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
