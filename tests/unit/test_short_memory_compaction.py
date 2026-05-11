from adapters import LLMResponse, Message, ToolCall, ToolResult
from agent import Agent


class CompactLLM:
    def __init__(self, response_text="压缩后的摘要"):
        self.response_text = response_text
        self.calls = []

    def chat(self, messages, tools, system):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "system": system,
        })
        return LLMResponse(text=self.response_text)


def _agent(llm=None):
    agent = Agent.__new__(Agent)
    agent.llm = llm or CompactLLM()
    agent.history = []
    agent.history_summary = None
    agent.pending_write_calls = []
    return agent


def _add_turn(agent, idx, answer_size=20):
    agent.history.append(Message(role="user", text=f"问题{idx}：分析 6000{idx:02d}"))
    agent.history.append(Message(role="assistant", text=f"回答{idx}：" + "A" * answer_size))


def test_short_memory_does_not_compact_below_threshold(monkeypatch):
    monkeypatch.setenv("SHORT_MEMORY_MAX_CHARS", "10000")
    llm = CompactLLM()
    agent = _agent(llm)
    for i in range(3):
        _add_turn(agent, i)

    agent._maybe_compact_history()

    assert agent.history_summary is None
    assert len(agent.history) == 6
    assert llm.calls == []


def test_short_memory_compacts_old_turns_and_keeps_recent_window(monkeypatch):
    monkeypatch.setenv("SHORT_MEMORY_MAX_CHARS", "100")
    monkeypatch.setenv("SHORT_MEMORY_KEEP_TURNS", "4")
    monkeypatch.setenv("SHORT_MEMORY_SUMMARY_MAX_CHARS", "3000")
    llm = CompactLLM("保留用户偏好：偏高股息；早期讨论过600000。")
    agent = _agent(llm)
    for i in range(6):
        _add_turn(agent, i, answer_size=80)

    agent._maybe_compact_history()

    assert agent.history_summary == "保留用户偏好：偏高股息；早期讨论过600000。"
    assert [m.text for m in agent.history if m.role == "user" and m.text] == [
        "问题2：分析 600002",
        "问题3：分析 600003",
        "问题4：分析 600004",
        "问题5：分析 600005",
    ]
    assert "问题0" in llm.calls[0]["messages"][0].text
    assert llm.calls[0]["tools"] == []


def test_history_for_llm_prepends_summary_without_mutating_history():
    agent = _agent()
    agent.history_summary = "用户偏好高股息低估值，关注600900。"
    _add_turn(agent, 1)

    view = agent._history_for_llm()

    assert view[0].role == "user"
    assert view[0].text.startswith("【此前对话压缩摘要】")
    assert "关注600900" in view[0].text
    assert view[1:] == agent.history
    assert agent.history[0].text == "问题1：分析 600001"


def test_short_memory_keeps_tool_call_and_result_together(monkeypatch):
    monkeypatch.setenv("SHORT_MEMORY_MAX_CHARS", "100")
    monkeypatch.setenv("SHORT_MEMORY_KEEP_TURNS", "4")
    llm = CompactLLM()
    agent = _agent(llm)
    _add_turn(agent, 0, answer_size=200)
    agent.history.extend([
        Message(role="user", text="问题1：看一下持仓"),
        Message(role="assistant", tool_calls=[
            ToolCall(id="call_1", name="get_positions", input={}),
        ]),
        Message(role="user", tool_results=[
            ToolResult(tool_call_id="call_1", content='[{"stock_code":"600900"}]'),
        ]),
        Message(role="assistant", text="持仓里有长江电力。"),
    ])
    for i in range(2, 5):
        _add_turn(agent, i, answer_size=80)

    agent._maybe_compact_history()

    kept_tool_call_ids = {
        tc.id
        for m in agent.history
        for tc in m.tool_calls
    }
    kept_tool_result_ids = {
        r.tool_call_id
        for m in agent.history
        for r in m.tool_results
    }
    assert "问题1：看一下持仓" in [m.text for m in agent.history if m.text]
    assert kept_tool_result_ids <= kept_tool_call_ids


def test_short_memory_merges_existing_summary(monkeypatch):
    monkeypatch.setenv("SHORT_MEMORY_MAX_CHARS", "100")
    monkeypatch.setenv("SHORT_MEMORY_KEEP_TURNS", "4")
    llm = CompactLLM("新旧合并后的摘要")
    agent = _agent(llm)
    agent.history_summary = "旧摘要：用户只接受低估值买入。"
    for i in range(6):
        _add_turn(agent, i, answer_size=80)

    agent._maybe_compact_history()

    assert agent.history_summary == "新旧合并后的摘要"
    assert "旧摘要：用户只接受低估值买入。" in llm.calls[0]["messages"][0].text


def test_reset_clears_short_memory_summary():
    agent = _agent()
    _add_turn(agent, 1)
    agent.history_summary = "旧摘要"
    agent.pending_write_calls = [ToolCall(id="call_1", name="upsert_position", input={})]

    agent.reset()

    assert agent.history == []
    assert agent.history_summary is None
    assert agent.pending_write_calls == []
