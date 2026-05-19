import json
import pytest
from types import SimpleNamespace

import pandas as pd

from adapters import LLMTimeoutError, Message, OpenAIAdapter
from agent import Agent
from tools import (
    ToolExecutor,
    calculate_dividend_reinvestment_projection,
    resolve_stock_code,
)


def test_projection_reinvests_only_whole_lots_and_carries_cash():
    out = calculate_dividend_reinvestment_projection(
        stock_code="601398",
        stock_name="工商银行",
        price=5.0,
        annual_dividend_per_share=0.3,
        monthly_lots=3,
        years=2,
        dividend_reinvest=True,
        dividend_source_date="2025-07-01",
    )

    assert out["regular_buy_shares"] == 7200
    assert out["reinvested_shares"] == 600
    assert out["ending_shares"] == 7800
    assert out["cumulative_dividend"] == 3300.0
    assert out["last_year_dividend"] == 2220.0
    assert out["remaining_cash"] == 300.0


def test_resolve_stock_code_maps_icbc_name():
    assert resolve_stock_code("每月定投工商银行3手") == "601398"
    assert resolve_stock_code("把伊利加入自选") == "600887"
    assert resolve_stock_code("定投 600028 2手") == "600028"


def test_tool_calculate_dividend_reinvestment(monkeypatch):
    monkeypatch.setattr("tools.fetch_tencent_quote", lambda stock_code: {
        "name": "工商银行",
        "price": 5.0,
        "pe_ttm": "5",
        "pb": "0.5",
        "market_cap_bn": "1",
        "52w_high": "6",
        "52w_low": "4",
    })
    dividend_df = pd.DataFrame([
        {"除权除息日": "2025-07-01", "派息": "3.0"},
        {"除权除息日": "2024-07-01", "派息": "2.5"},
    ])
    monkeypatch.setitem(
        __import__("sys").modules,
        "akshare",
        SimpleNamespace(stock_history_dividend_detail=lambda **kwargs: dividend_df),
    )

    executor = ToolExecutor(memory=None)
    out = json.loads(executor.execute("calculate_dividend_reinvestment", {
        "stock_code": "601398",
        "monthly_lots": 3,
        "years": 2,
        "dividend_reinvest": True,
    }))

    assert out["stock_code"] == "601398"
    assert out["stock_name"] == "工商银行"
    assert out["annual_dividend_per_share"] == 0.3
    assert out["dividend_source_date"] == "2025-07-01"
    assert out["cumulative_dividend"] == 3300.0


def test_get_stock_data_does_not_sync_compute_pe_percentile_by_default(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "investment.db"))
    monkeypatch.setattr("tools.fetch_tencent_quote", lambda stock_code: {
        "name": "工商银行",
        "price": 5.0,
        "pe_ttm": "5",
        "pb": "0.5",
        "market_cap_bn": "1",
        "52w_high": "6",
        "52w_low": "4",
    })
    dividend_df = pd.DataFrame([
        {"除权除息日": "2026-01-01", "派息": "3.0"},
    ])
    monkeypatch.setitem(
        __import__("sys").modules,
        "akshare",
        SimpleNamespace(stock_history_dividend_detail=lambda **kwargs: dividend_df),
    )

    def fail_sync_compute(*args, **kwargs):
        raise AssertionError("PE percentile should not be computed synchronously")

    warmed = []
    monkeypatch.setattr(ToolExecutor, "_compute_pe_percentile", fail_sync_compute)
    monkeypatch.setattr(
        ToolExecutor,
        "_warm_pe_percentile_cache_async",
        classmethod(lambda cls, stock_code, current_pe: warmed.append((stock_code, current_pe))),
    )

    executor = ToolExecutor(memory=None)
    out = json.loads(executor.execute("get_stock_data", {"stock_code": "601398"}))

    assert out["stock_code"] == "601398"
    assert out["ttm_yield_pct"] == 6.0
    assert out["pe_percentile_pct"] is None
    assert "后台缓存" in out["pe_percentile_note"]
    assert warmed == [("601398", 5.0)]


def test_agent_directly_answers_dividend_reinvestment_without_llm():
    class FailingLLM:
        def chat(self, *args, **kwargs):
            raise AssertionError("LLM should not be called")

    class FakeExecutor:
        def execute(self, tool_name, tool_input, allow_write=False):
            assert tool_name == "calculate_dividend_reinvestment"
            assert tool_input == {
                "stock_code": "601398",
                "monthly_lots": 3,
                "years": 30,
                "dividend_reinvest": True,
            }
            return json.dumps({
                "stock_code": "601398",
                "stock_name": "工商银行",
                "price": 5.0,
                "annual_dividend_per_share": 0.3,
                "dividend_source_date": "2025-07-01",
                "monthly_lots": 3,
                "monthly_shares": 300,
                "years": 30,
                "dividend_reinvest": True,
                "regular_buy_shares": 108000,
                "reinvested_shares": 120000,
                "ending_shares": 228000,
                "cumulative_dividend": 500000.0,
                "last_year_dividend": 68000.0,
                "remaining_cash": 100.0,
                "assumption": "测试假设。",
                "yearly": [
                    {"year": 28, "cash_dividend": 60000.0, "reinvested_shares": 12000, "ending_shares": 200000},
                    {"year": 29, "cash_dividend": 64000.0, "reinvested_shares": 12800, "ending_shares": 215000},
                    {"year": 30, "cash_dividend": 68000.0, "reinvested_shares": 13600, "ending_shares": 228000},
                ],
            }, ensure_ascii=False)

    agent = Agent.__new__(Agent)
    agent.llm = FailingLLM()
    agent.executor = FakeExecutor()
    agent.history = []
    agent.pending_write_calls = []

    out = agent.chat("为我计算一下 如果每个月定投工商银行3手 股息复投 三十年后 我的股息总收益是多少 当年的股息是多少")

    assert "30年累计股息" in out
    assert "第30年当年股息" in out
    assert "500,000.00 元" in out


def test_openai_adapter_uses_explicit_httpx_timeout():
    adapter = OpenAIAdapter(
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key="test-key",
        timeout=60,
        max_retries=0,
    )

    assert adapter.timeout.connect == 10.0
    assert adapter.timeout.read == 60.0
    assert adapter.timeout.write == 10.0
    assert adapter.timeout.pool == 10.0


def test_openai_adapter_wraps_provider_timeout():
    adapter = OpenAIAdapter(
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key="test-key",
        timeout=180,
        max_retries=0,
    )

    class TimeoutCompletions:
        def create(self, **kwargs):
            raise TimeoutError("Request timed out.")

    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=TimeoutCompletions(),
        ),
    )

    with pytest.raises(LLMTimeoutError, match="180 seconds"):
        adapter.chat(
            messages=[Message(role="user", text="hello")],
            tools=[],
            system="system",
        )


def test_openai_adapter_streams_text_chunks():
    adapter = OpenAIAdapter(
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key="test-key",
        timeout=180,
        max_retries=0,
    )

    class StreamingCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return iter([
                SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content="分批"),
                        finish_reason=None,
                    )],
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content="输出"),
                        finish_reason=None,
                    )],
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content=None),
                        finish_reason="stop",
                    )],
                ),
            ])

    completions = StreamingCompletions()
    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )

    chunks = list(adapter.chat_stream(
        messages=[Message(role="user", text="hello")],
        tools=[],
        system="system",
    ))

    assert completions.kwargs["stream"] is True
    assert [chunk.text_delta for chunk in chunks if chunk.text_delta] == ["分批", "输出"]
    assert chunks[-1].response.text == "分批输出"
