import json
import time
from threading import Event

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
        if not tools:
            return LLMResponse(text="")
        return self.response


def _make_agent(llm, memory):
    agent = Agent.__new__(Agent)
    agent.llm = llm
    agent.memory = memory
    agent.executor = ToolExecutor(memory)
    agent.max_steps = 5
    agent.history = []
    agent.pending_write_calls = []
    agent._save_conversation_insight = lambda *args: None
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


def test_async_memory_write_does_not_block_chat(monkeypatch):
    monkeypatch.setenv("ASYNC_MEMORY_WRITE", "true")
    decisions = FakeDecisions()
    memory = FakeMemory(decisions)
    llm = FakeLLM(LLMResponse(text="ok"))
    agent = _make_agent(llm, memory)
    saved = Event()

    def slow_save(*args):
        time.sleep(0.2)
        saved.set()

    agent._save_conversation_insight = slow_save

    start = time.monotonic()
    out = agent.chat("比较 600900 和 601088 的 PE 股息率")
    elapsed = time.monotonic() - start

    assert out == "ok"
    assert elapsed < 0.15
    assert saved.wait(1)
