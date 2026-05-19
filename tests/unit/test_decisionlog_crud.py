"""DecisionLog CRUD 测试：用 :memory: SQLite，毫秒级跑完。"""

import sqlite3

import pytest

from memory import DecisionLog


def test_save_and_search_decision(fresh_decisionlog):
    log = fresh_decisionlog
    new_id = log.save_decision({
        "stock_code": "600028",
        "stock_name": "中国石化",
        "action": "watch",
        "view": "bullish",
        "reasoning": "高股息低估值",
        "price": 6.5,
        "ttm_yield": 5.2,
        "pe_pct": 15.0,
    })
    assert new_id > 0

    rows = log.search_decisions(stock_code="600028")
    assert len(rows) == 1
    assert rows[0]["stock_name"] == "中国石化"
    assert rows[0]["view"] == "bullish"


def test_search_by_keyword(fresh_decisionlog):
    log = fresh_decisionlog
    log.save_decision({
        "stock_code": "600028", "stock_name": "中国石化",
        "reasoning": "防御型核心仓",
    })
    log.save_decision({
        "stock_code": "300750", "stock_name": "宁德时代",
        "reasoning": "成长股高波动",
    })

    rows = log.search_decisions(keyword="防御")
    assert len(rows) == 1
    assert rows[0]["stock_code"] == "600028"


def test_decisions_are_append_only_via_delete(fresh_decisionlog):
    """delete_decision 工作，但没有 update —— append-only 设计的边界。"""
    log = fresh_decisionlog
    new_id = log.save_decision({"stock_code": "600028", "reasoning": "t"})
    assert log.delete_decision(new_id) is True
    assert log.delete_decision(new_id) is False  # 二次删除返回 False


def test_position_upsert_and_delete(fresh_decisionlog):
    log = fresh_decisionlog
    log.upsert_position({
        "stock_code": "600028", "stock_name": "中国石化",
        "cost_price": 6.0, "shares": 1000, "position_pct": 30, "tier": "core",
    })
    positions = log.get_positions()
    assert len(positions) == 1
    assert positions[0]["cost_price"] == 6.0
    assert positions[0]["tier"] == "core"

    # upsert 同 code → 更新而非新增
    log.upsert_position({
        "stock_code": "600028", "stock_name": "中国石化",
        "cost_price": 6.5, "tier": "core",
    })
    positions = log.get_positions()
    assert len(positions) == 1
    assert positions[0]["cost_price"] == 6.5

    assert log.delete_position("600028") is True
    assert log.get_positions() == []


def test_watchlist_upsert_and_delete(fresh_decisionlog):
    log = fresh_decisionlog
    log.upsert_watchlist({
        "stock_code": "601398", "stock_name": "工商银行",
        "reason": "高股息防御", "alert_yield": 6.0, "alert_pe_pct": 20,
        "alert_price_below": 4.8, "watch_price_below": 5.1,
        "alert_note": "等年报确认ROE",
    })
    items = log.get_watchlist()
    assert len(items) == 1
    assert items[0]["alert_yield"] == 6.0
    assert items[0]["alert_price_below"] == 4.8
    assert items[0]["watch_price_below"] == 5.1
    assert items[0]["alert_note"] == "等年报确认ROE"

    log.delete_watchlist("601398")
    assert log.get_watchlist() == []


def test_retrospective_attaches_to_decision(fresh_decisionlog):
    log = fresh_decisionlog
    decision_id = log.save_decision({
        "stock_code": "600028", "reasoning": "看多",
    })
    retro_id = log.save_retrospective({
        "decision_id": decision_id,
        "price_now": 7.2,
        "outcome": "correct",
        "what_i_missed": None,
        "updated_view": "继续持有",
    })
    assert retro_id > 0

    retros = log.search_retrospectives(decision_id)
    assert len(retros) == 1
    assert retros[0]["outcome"] == "correct"


def test_scan_result_save(fresh_decisionlog):
    log = fresh_decisionlog
    rid = log.save_scan_result({
        "scope": "positions",
        "stock_code": "600028",
        "stock_name": "中国石化",
        "signal": "alert",
        "summary": "现价6.5 | 跌幅12%",
    })
    assert rid > 0

    row = log.conn.execute("SELECT * FROM scan_results WHERE id=?", (rid,)).fetchone()
    assert row["signal"] == "alert"


def test_decisionlog_init_error_includes_sqlite_path_diagnostics(monkeypatch, tmp_path):
    class BrokenConn:
        row_factory = None

        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        def executescript(self, *args, **kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        def close(self):
            pass

    monkeypatch.setattr("memory.sqlite3.connect", lambda *args, **kwargs: BrokenConn())

    db_path = tmp_path / "investment_live.db"
    with pytest.raises(RuntimeError) as exc:
        DecisionLog(db_path=str(db_path))

    msg = str(exc.value)
    assert "SQLite数据库初始化失败" in msg
    assert "unable to open database file" in msg
    assert "db_path=" in msg
    assert "parent_exists=True" in msg
