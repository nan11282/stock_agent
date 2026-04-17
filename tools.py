"""
tools.py -- 工具定义 + 执行器

READ  tools (9) : Agent 可自主调用，无需请示
WRITE tools (7) : 必须用户明确指令 + 展示内容 + 等待"确认"后才调用
"""

import json
from memory import MemoryManager


# ─────────────────────────────────────────────
# Tool Schema（传给 LLM 的格式）
# ─────────────────────────────────────────────

READ_TOOLS = [
    {
        "name": "get_stock_data",
        "description": "获取A股实时行情与估值：当前价格、PE、PB、TTM股息率。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "6位股票代码，如 600028"},
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "get_dividend_history",
        "description": "获取历史分红记录：近N年每年每股分红和股息率。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string"},
                "years": {"type": "integer", "description": "查多少年，默认5"},
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "get_financials",
        "description": "获取财务摘要：营收、净利润、ROE、EPS、派息率、负债率。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string"},
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "get_ah_premium",
        "description": "查询AH股溢价率（仅适用AH两地上市公司）。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "A股代码"},
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "search_decisions",
        "description": "在历史决策日志中搜索记录，用于复盘或查找过去分析。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string"},
                "keyword": {"type": "string"},
                "limit": {"type": "integer", "description": "最多返回多少条，默认10"},
            },
        },
    },
    {
        "name": "get_positions",
        "description": "查询当前所有持仓股票，含成本价、仓位比例、分层。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_watchlist",
        "description": "查询自选股关注列表，含关注原因和提醒阈值。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "search_retrospectives",
        "description": "查询某条决策的所有复盘记录。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision_id": {"type": "integer", "description": "决策记录ID"},
            },
            "required": ["decision_id"],
        },
    },
    {
        "name": "retrieve_memory",
        "description": "从向量+FTS5混合记忆库中语义检索相关历史洞察。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "description": "返回条数，默认4"},
            },
            "required": ["query"],
        },
    },
]

WRITE_TOOLS = [
    {
        "name": "save_decision",
        "description": (
            "将投资决策持久化到数据库。\n"
            "【严格限制】满足以下两个条件才能调用：\n"
            '  1. 用户明确说"保存"、"存下来"、"记录这个"等明确指令\n'
            '  2. 已向用户展示将要保存的结构化内容，用户回复"确认"\n'
            "分析过程中绝对禁止自主调用此工具。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code":  {"type": "string"},
                "stock_name":  {"type": "string"},
                "action":      {"type": "string",
                                "enum": ["buy_signal", "sell_signal", "hold", "watch", "analysis"]},
                "view":        {"type": "string",
                                "enum": ["bullish", "bearish", "neutral"]},
                "reasoning":   {"type": "string", "description": "完整逻辑，不压缩"},
                "price":       {"type": "number"},
                "ttm_yield":   {"type": "number"},
                "pe_pct":      {"type": "number", "description": "PE历史百分位 0-100"},
                "pe_abs":      {"type": "number"},
                "tags":        {"type": "array", "items": {"type": "string"}},
            },
            "required": ["stock_code", "reasoning"],
        },
    },
    {
        "name": "delete_decision",
        "description": (
            "删除指定id的决策记录。\n"
            "【严格限制】满足以下两个条件才能调用：\n"
            '  1. 用户明确说"删掉"、"删除这条"、"移除"等明确指令\n'
            '  2. 已向用户展示将被删除的记录内容，用户回复"确认"'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "decision_id": {"type": "integer"},
            },
            "required": ["decision_id"],
        },
    },
    {
        "name": "save_retrospective",
        "description": (
            "保存复盘记录，挂在原始决策下，原始记录不修改。\n"
            "【严格限制】满足以下两个条件才能调用：\n"
            '  1. 用户明确说"保存复盘"、"记录复盘"等明确指令\n'
            '  2. 已向用户展示将要保存的复盘内容，用户回复"确认"'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "decision_id":   {"type": "integer"},
                "price_now":     {"type": "number"},
                "outcome":       {"type": "string", "enum": ["correct", "wrong", "partial"]},
                "what_i_missed": {"type": "string"},
                "updated_view":  {"type": "string"},
            },
            "required": ["decision_id", "outcome"],
        },
    },
    {
        "name": "upsert_position",
        "description": (
            "新增或更新持仓记录。\n"
            "【严格限制】满足以下两个条件才能调用：\n"
            '  1. 用户明确说"加入持仓"、"更新持仓"、"记录买入"等明确指令\n'
            '  2. 已向用户展示将要保存的持仓内容，用户回复"确认"'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code":   {"type": "string"},
                "stock_name":   {"type": "string"},
                "cost_price":   {"type": "number", "description": "成本价"},
                "shares":       {"type": "integer", "description": "持仓股数"},
                "position_pct": {"type": "number", "description": "仓位百分比 0-100"},
                "tier":         {"type": "string", "enum": ["core", "growth"],
                                 "description": "core=核心仓 growth=成长仓"},
            },
            "required": ["stock_code", "stock_name", "cost_price"],
        },
    },
    {
        "name": "delete_position",
        "description": (
            "从持仓中移除某只股票。\n"
            "【严格限制】满足以下两个条件才能调用：\n"
            '  1. 用户明确说"清仓"、"移除持仓"、"删除持仓"等明确指令\n'
            '  2. 已向用户展示将被移除的持仓信息，用户回复"确认"'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string"},
            },
            "required": ["stock_code"],
        },
    },
    {
        "name": "upsert_watchlist",
        "description": (
            "将股票加入或更新自选股关注列表。\n"
            "【严格限制】满足以下两个条件才能调用：\n"
            '  1. 用户明确说"加入自选"、"关注这只"、"加到观察列表"等明确指令\n'
            '  2. 已向用户展示将要加入的自选股信息，用户回复"确认"'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code":    {"type": "string"},
                "stock_name":    {"type": "string"},
                "reason":        {"type": "string", "description": "关注原因"},
                "alert_yield":   {"type": "number", "description": "股息率触发阈值(%)，达到即提醒"},
                "alert_pe_pct":  {"type": "number", "description": "PE百分位触发阈值，低于即提醒"},
            },
            "required": ["stock_code", "stock_name"],
        },
    },
    {
        "name": "delete_watchlist",
        "description": (
            "从自选股列表中移除某只股票。\n"
            "【严格限制】满足以下两个条件才能调用：\n"
            '  1. 用户明确说"移出自选"、"不再关注"、"删除自选"等明确指令\n'
            '  2. 已向用户展示将被移除的自选股信息，用户回复"确认"'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string"},
            },
            "required": ["stock_code"],
        },
    },
]

ALL_TOOLS = READ_TOOLS + WRITE_TOOLS


# ─────────────────────────────────────────────
# Tool Executor
# ─────────────────────────────────────────────

class ToolExecutor:
    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def execute(self, tool_name: str, tool_input: dict) -> str:
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)
            result = handler(**tool_input)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": tool_name}, ensure_ascii=False)

    # ── READ handlers ─────────────────────────

    def _tool_get_stock_data(self, stock_code: str) -> dict:
        import akshare as ak

        info_df = ak.stock_individual_info_em(symbol=stock_code)
        info = dict(zip(info_df["item"], info_df["value"]))

        price = float(info.get("最新", 0) or 0)
        name = info.get("股票简称", "")

        ttm_yield = None
        try:
            div_df = ak.stock_history_dividend_detail(
                symbol=stock_code, indicator="分红"
            )
            if not div_df.empty and price > 0:
                import pandas as pd
                div_df["除权除息日"] = pd.to_datetime(
                    div_df["除权除息日"], errors="coerce"
                )
                cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
                recent = div_df[div_df["除权除息日"] >= cutoff]
                if not recent.empty:
                    total_div_per_10 = recent["派息(每10股税前)"].astype(float).sum()
                    div_per_share = total_div_per_10 / 10
                    ttm_yield = round(div_per_share / price * 100, 2)
        except Exception:
            pass

        return {
            "stock_code":    stock_code,
            "name":          name,
            "price":         price,
            "pe_ttm":        info.get("市盈率(动态)"),
            "pb":            info.get("市净率"),
            "market_cap_bn": info.get("总市值"),
            "ttm_yield_pct": ttm_yield,
            "52w_high":      info.get("52周最高"),
            "52w_low":       info.get("52周最低"),
        }

    def _tool_get_dividend_history(self, stock_code: str, years: int = 5) -> list:
        import akshare as ak
        import pandas as pd

        df = ak.stock_history_dividend_detail(symbol=stock_code, indicator="分红")
        if df.empty:
            return []

        df["除权除息日"] = pd.to_datetime(df["除权除息日"], errors="coerce")
        df = df.dropna(subset=["除权除息日"])

        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        df = df[df["除权除息日"] >= cutoff].copy()
        df = df.sort_values("除权除息日", ascending=False)

        records = []
        for _, row in df.iterrows():
            try:
                div_per_share = float(row["派息(每10股税前)"]) / 10
            except (ValueError, TypeError):
                div_per_share = None

            records.append({
                "date":            str(row["除权除息日"].date()),
                "div_per_share":   div_per_share,
                "bonus_per_10":    row.get("送股(每10股)"),
                "transfer_per_10": row.get("转增(每10股)"),
            })

        return records

    def _tool_get_financials(self, stock_code: str) -> list:
        import akshare as ak

        df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按年度")
        if df.empty:
            return []

        df = df.head(5)

        keep_cols = [
            "报告期", "营业总收入", "归母净利润", "每股收益",
            "净资产收益率", "每股净资产", "资产负债率",
        ]
        existing = [c for c in keep_cols if c in df.columns]
        df = df[existing]

        return df.to_dict(orient="records")

    def _tool_get_ah_premium(self, stock_code: str) -> dict:
        import akshare as ak

        df = ak.stock_zh_ah_spot_em()
        if df.empty:
            return {"error": "AH 数据获取失败"}

        code_col = next((c for c in df.columns if "代码" in c and "A" in c), None)
        if code_col is None:
            code_col = df.columns[0]

        row = df[df[code_col] == stock_code]
        if row.empty:
            return {
                "stock_code": stock_code,
                "note": "该股票不在AH比价列表中，可能未在港股上市",
            }

        row = row.iloc[0]
        return {
            "stock_code":  stock_code,
            "a_price":     row.get("A股价格") or row.get("A股最新价"),
            "h_price_hkd": row.get("H股价格") or row.get("H股最新价"),
            "premium_pct": row.get("AH股溢价率"),
            "h_code":      row.get("H股代码"),
        }

    def _tool_search_decisions(self, stock_code: str = None,
                               keyword: str = None, limit: int = 10) -> list:
        return self.memory.decisions.search_decisions(
            stock_code=stock_code, keyword=keyword, limit=limit
        )

    def _tool_get_positions(self) -> list:
        return self.memory.decisions.get_positions()

    def _tool_get_watchlist(self) -> list:
        return self.memory.decisions.get_watchlist()

    def _tool_search_retrospectives(self, decision_id: int) -> list:
        return self.memory.decisions.search_retrospectives(decision_id)

    def _tool_retrieve_memory(self, query: str, top_k: int = 4) -> list:
        return self.memory.episodic.retrieve(query, top_k=top_k)

    # ── WRITE handlers ────────────────────────

    def _tool_save_decision(self, **data) -> dict:
        new_id = self.memory.decisions.save_decision(data)
        return {"status": "saved", "id": new_id}

    def _tool_delete_decision(self, decision_id: int) -> dict:
        success = self.memory.decisions.delete_decision(decision_id)
        return {"status": "deleted" if success else "not_found", "id": decision_id}

    def _tool_save_retrospective(self, **data) -> dict:
        new_id = self.memory.decisions.save_retrospective(data)
        return {"status": "saved", "id": new_id}

    def _tool_upsert_position(self, **data) -> dict:
        self.memory.decisions.upsert_position(data)
        return {"status": "saved", "stock_code": data["stock_code"]}

    def _tool_delete_position(self, stock_code: str) -> dict:
        success = self.memory.decisions.delete_position(stock_code)
        return {"status": "deleted" if success else "not_found", "stock_code": stock_code}

    def _tool_upsert_watchlist(self, **data) -> dict:
        self.memory.decisions.upsert_watchlist(data)
        return {"status": "saved", "stock_code": data["stock_code"]}

    def _tool_delete_watchlist(self, stock_code: str) -> dict:
        success = self.memory.decisions.delete_watchlist(stock_code)
        return {"status": "deleted" if success else "not_found", "stock_code": stock_code}
