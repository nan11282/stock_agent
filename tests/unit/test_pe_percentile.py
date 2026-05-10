"""PE 历史百分位测试：直接喂 fin_df / price_df，不打 akshare。"""

import pandas as pd
import pytest

from tools import ToolExecutor


def _fin_df(years_eps: list[tuple[int, float]]) -> pd.DataFrame:
    """构造一个 stock_financial_abstract_ths 风格的 fin_df。"""
    return pd.DataFrame([
        {"报告期": str(y), "基本每股收益": eps} for y, eps in years_eps
    ])


def _price_df(year_close_pairs: list[tuple[int, float]]) -> pd.DataFrame:
    """构造每年最后一日的收盘价。"""
    rows = []
    for year, close in year_close_pairs:
        rows.append({"date": f"{year}-12-31", "close": close})
    return pd.DataFrame(rows)


def test_basic_percentile_when_current_pe_low():
    """历史 PE: [10,12,14,16,18,20]，当前 PE=11 → 1/6 ≈ 16.7%。"""
    eps_pairs = [(2018, 1.0), (2019, 1.0), (2020, 1.0),
                 (2021, 1.0), (2022, 1.0), (2023, 1.0)]
    price_pairs = [(2018, 10), (2019, 12), (2020, 14),
                   (2021, 16), (2022, 18), (2023, 20)]
    result = ToolExecutor._compute_pe_percentile(
        "600028", current_pe=11.0,
        fin_df=_fin_df(eps_pairs), price_df=_price_df(price_pairs),
    )
    assert result["pe_percentile_pct"] == pytest.approx(16.7, abs=0.1)
    assert result["pe_percentile_years"] == 6
    assert result["pe_percentile_lo"] == 10.0
    assert result["pe_percentile_hi"] == 20.0


def test_percentile_when_current_pe_highest():
    eps_pairs = [(y, 1.0) for y in range(2018, 2024)]
    price_pairs = [(2018, 10), (2019, 12), (2020, 14),
                   (2021, 16), (2022, 18), (2023, 20)]
    result = ToolExecutor._compute_pe_percentile(
        "600028", current_pe=100.0,
        fin_df=_fin_df(eps_pairs), price_df=_price_df(price_pairs),
    )
    # 全部历史都 < 当前 → 100%
    assert result["pe_percentile_pct"] == 100.0


def test_percentile_when_current_pe_lowest():
    eps_pairs = [(y, 1.0) for y in range(2018, 2024)]
    price_pairs = [(2018, 10), (2019, 12), (2020, 14),
                   (2021, 16), (2022, 18), (2023, 20)]
    result = ToolExecutor._compute_pe_percentile(
        "600028", current_pe=1.0,
        fin_df=_fin_df(eps_pairs), price_df=_price_df(price_pairs),
    )
    # 没有历史 < 当前 → 0%
    assert result["pe_percentile_pct"] == 0.0


def test_skip_negative_eps_year():
    """亏损年份（EPS<=0）应被跳过。"""
    eps_pairs = [(2018, 1.0), (2019, -0.5), (2020, 1.0),
                 (2021, 1.0), (2022, 1.0), (2023, 1.0)]
    price_pairs = [(y, 10 + i) for i, y in enumerate(range(2018, 2024))]
    result = ToolExecutor._compute_pe_percentile(
        "600028", current_pe=10.0,
        fin_df=_fin_df(eps_pairs), price_df=_price_df(price_pairs),
    )
    # 6 年里有 1 年亏损 → 5 年有效
    assert result["pe_percentile_years"] == 5


def test_too_few_years_returns_note():
    eps_pairs = [(2022, 1.0), (2023, 1.0)]
    price_pairs = [(2022, 10), (2023, 12)]
    result = ToolExecutor._compute_pe_percentile(
        "600028", current_pe=10.0,
        fin_df=_fin_df(eps_pairs), price_df=_price_df(price_pairs),
    )
    assert result["pe_percentile_pct"] is None
    assert "≥3年" in result["pe_percentile_note"]


def test_empty_fin_df():
    result = ToolExecutor._compute_pe_percentile(
        "600028", current_pe=10.0,
        fin_df=pd.DataFrame(), price_df=_price_df([(2023, 10)]),
    )
    assert result["pe_percentile_pct"] is None


def test_empty_price_df():
    eps_pairs = [(y, 1.0) for y in range(2018, 2024)]
    result = ToolExecutor._compute_pe_percentile(
        "600028", current_pe=10.0,
        fin_df=_fin_df(eps_pairs), price_df=pd.DataFrame(),
    )
    assert result["pe_percentile_pct"] is None


def test_cached_pe_history_recomputes_percentile_for_current_pe(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "investment.db"))
    pe_info = {
        "pe_percentile_pct": 0.0,
        "pe_percentile_years": 4,
        "pe_percentile_lo": 10.0,
        "pe_percentile_hi": 40.0,
        "pe_history": [10.0, 20.0, 30.0, 40.0],
    }

    ToolExecutor._init_pe_cache()
    ToolExecutor._save_cached_pe_history("600028", pe_info)
    cached = ToolExecutor._load_cached_pe_percentile("600028", current_pe=35.0)

    assert cached["pe_percentile_cached"] is True
    assert cached["pe_percentile_pct"] == 75.0


def test_compact_pe_percentile_removes_history_payload():
    compact = ToolExecutor._compact_pe_percentile({
        "pe_percentile_pct": 50.0,
        "pe_percentile_years": 3,
        "pe_percentile_lo": 8.0,
        "pe_percentile_hi": 16.0,
        "pe_history": [8.0, 12.0, 16.0],
    })

    assert "pe_history" not in compact
    assert compact["pe_percentile_range"] == {"low": 8.0, "high": 16.0, "years": 3}
