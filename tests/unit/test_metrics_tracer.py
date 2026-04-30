"""Tracer 自身的单测：保证埋点结构和落盘格式不被回归弄坏。"""

import json
import os

import pytest

from metrics import (
    LLMCallRecord,
    ToolCallRecord,
    Tracer,
    TurnRecord,
    history_chars,
)
from adapters import Message, ToolCall, ToolResult


def test_disabled_tracer_is_noop(tmp_path):
    t = Tracer(trace_dir=str(tmp_path), enabled=False)
    with t.turn("hi", []):
        with t.llm_call("sys", []) as rec:
            rec.output_text_chars = 99   # 应静默丢弃，不抛错
        with t.tool_call("foo", {"a": 1}) as tc:
            tc.result_chars = 100
    # 没有 jsonl 落盘
    assert list(tmp_path.iterdir()) == []


def test_enabled_tracer_dumps_jsonl(tmp_path):
    t = Tracer(trace_dir=str(tmp_path), enabled=True)
    with t.turn("user query", history_snapshot=[]):
        t.set_system_prompt("system prompt 12345")
        with t.llm_call("system prompt 12345", []) as rec:
            rec.output_text_chars = 30
            rec.tool_calls_emitted = 1
        with t.tool_call("get_stock_data", {"stock_code": "600028"}) as tc:
            tc.result_chars = 512
        t.set_history_end([], "final response text")

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text(encoding="utf-8").strip())

    assert rec["user_input_chars"] == len("user query")
    assert rec["system_prompt_chars"] == len("system prompt 12345")
    assert rec["react_steps"] == 1
    assert len(rec["llm_calls"]) == 1
    assert rec["llm_calls"][0]["output_text_chars"] == 30
    assert rec["llm_calls"][0]["tool_calls_emitted"] == 1
    assert len(rec["tool_calls"]) == 1
    assert rec["tool_calls"][0]["name"] == "get_stock_data"
    assert rec["tool_calls"][0]["result_chars"] == 512
    assert rec["final_response_chars"] == len("final response text")


def test_history_chars_counts_text_and_tool_calls():
    history = [
        Message(role="user", text="hello"),                        # 5
        Message(role="assistant", text="hi", tool_calls=[          # 2 + name "foo"=3 + json "{\"k\": \"v\"}"
            ToolCall(id="1", name="foo", input={"k": "v"})
        ]),
        Message(role="user", tool_results=[                         # content 4
            ToolResult(tool_call_id="1", content="data")
        ]),
    ]
    n = history_chars(history)
    # 不在于精确值，而是单调性：≥ 各部分明显字符数之和
    assert n >= len("hello") + len("hi") + len("foo") + len("data")


def test_react_steps_increments_per_llm_call(tmp_path):
    t = Tracer(trace_dir=str(tmp_path), enabled=True)
    with t.turn("q", []):
        for _ in range(3):
            with t.llm_call("sys", []):
                pass
    rec = json.loads((tmp_path / list(os.listdir(tmp_path))[0]).read_text(encoding="utf-8"))
    assert rec["react_steps"] == 3
    assert [c["step"] for c in rec["llm_calls"]] == [1, 2, 3]


def test_tool_call_error_recorded(tmp_path):
    t = Tracer(trace_dir=str(tmp_path), enabled=True)
    with t.turn("q", []):
        with pytest.raises(ValueError):
            with t.tool_call("bad_tool", {}) as tc:
                raise ValueError("boom")
    rec = json.loads((tmp_path / list(os.listdir(tmp_path))[0]).read_text(encoding="utf-8"))
    assert len(rec["tool_calls"]) == 1
    assert "boom" in rec["tool_calls"][0]["error"]
