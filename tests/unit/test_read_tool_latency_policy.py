from adapters import ToolCall
from agent import Agent


def _stock_call(include_pe_percentile=True):
    return ToolCall(
        id="call_stock",
        name="get_stock_data",
        input={
            "stock_code": "600900",
            "include_pe_percentile": include_pe_percentile,
        },
    )


def test_stock_data_defaults_to_async_pe_percentile_for_fast_replies():
    calls = [_stock_call(include_pe_percentile=True)]

    guarded = Agent._guard_read_tool_calls(
        "对比 600900、600795、601985、600011 哪个更值得买",
        calls,
    )

    assert guarded[0].input["include_pe_percentile"] is False
    assert guarded[0].input["async_pe_percentile"] is True


def test_stock_data_keeps_sync_pe_percentile_when_user_explicitly_asks():
    calls = [_stock_call(include_pe_percentile=True)]

    guarded = Agent._guard_read_tool_calls(
        "给我 600900 的完整估值和 PE百分位",
        calls,
    )

    assert guarded[0].input["include_pe_percentile"] is True
    assert "async_pe_percentile" not in guarded[0].input


def test_default_read_tool_round_limit_is_one(monkeypatch):
    monkeypatch.delenv("AGENT_READ_TOOL_ROUNDS", raising=False)

    assert Agent._read_tool_round_limit() == 1
