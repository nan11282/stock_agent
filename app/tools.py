"""
tools.py -- 工具定义 + 执行器

READ  tools (9) : Agent 可自主调用，无需请示
WRITE tools (7) : 必须用户明确指令 + 展示内容 + 等待"确认"后才调用
"""

import json
from memory import MemoryManager
from metrics import get_tracer


# ─────────────────────────────────────────────
# Tool Schema（传给 LLM 的格式）
# ─────────────────────────────────────────────

READ_TOOLS = [
    {
        "name": "get_stock_data",
        "description": "获取A股实时行情与估值：当前价格、PE、PB、TTM股息率、PE历史百分位（当前PE在近10年中的分位，越低越便宜）。Agent可自主调用。",
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

# ── 共享工具函数 ──────────────────────────

def exchange_prefix(stock_code: str) -> str:
    """根据A股代码推断交易所前缀，返回 'sh' 或 'sz'。"""
    if stock_code.startswith(("6", "68")):
        return "sh"
    return "sz"


def fetch_tencent_quote(stock_code: str) -> dict:
    """从腾讯行情 API 获取单只股票实时数据。"""
    import requests
    prefix = exchange_prefix(stock_code)
    url = f"http://qt.gtimg.cn/q={prefix}{stock_code}"
    r = requests.get(url, timeout=10)
    r.encoding = "gbk"
    fields = r.text.split("~")
    return {
        "name":           fields[1],
        "price":          float(fields[3]) if fields[3] else 0.0,
        "pe_ttm":         fields[39],
        "pb":             fields[46],
        "market_cap_bn":  fields[45],
        "52w_high":       fields[47],
        "52w_low":        fields[48],
    }


class ToolExecutor:
    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def execute(self, tool_name: str, tool_input: dict) -> str:
        with get_tracer().tool_call(tool_name, tool_input) as rec:
            try:
                handler = getattr(self, f"_tool_{tool_name}", None)
                if handler is None:
                    out = json.dumps(
                        {"error": f"未知工具: {tool_name}"}, ensure_ascii=False
                    )
                else:
                    result = handler(**tool_input)
                    out = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception as e:
                out = json.dumps(
                    {"error": str(e), "tool": tool_name}, ensure_ascii=False
                )
                rec.error = str(e)
            rec.result_chars = len(out)
            return out

    # ── PE 历史百分位计算（内部复用）────────────

    @staticmethod
    def _compute_pe_percentile(stock_code: str, current_pe: float,
                               fin_df=None, price_df=None) -> dict:
        """计算当前 PE 在近 10 年历史中的分位。返回 pe_percentile_pct 等字段。

        测试时可直接传入 fin_df / price_df，跳过 akshare 调用。
        """
        import pandas as pd

        try:
            # 年度财务数据（EPS）
            if fin_df is None:
                import akshare as ak
                fin_df = ak.stock_financial_abstract_ths(
                    symbol=stock_code, indicator="按年度"
                )
            if fin_df.empty or "基本每股收益" not in fin_df.columns:
                return {"pe_percentile_pct": None, "pe_percentile_note": "无财务EPS数据"}

            # 10年日K线（腾讯源，Docker 内可用）
            if price_df is None:
                import akshare as ak
                end_date = pd.Timestamp.now().strftime("%Y%m%d")
                start_date = (pd.Timestamp.now() - pd.DateOffset(years=10)).strftime("%Y%m%d")
                prefix = exchange_prefix(stock_code)
                price_df = ak.stock_zh_a_hist_tx(
                    symbol=f"{prefix}{stock_code}", start_date=start_date,
                    end_date=end_date, adjust="qfq",
                )
            if price_df.empty:
                return {"pe_percentile_pct": None, "pe_percentile_note": "无历史价格数据"}

            price_df["date"] = pd.to_datetime(price_df["date"])

            # 每年报告期找最近交易日收盘价，计算该年 PE
            # 财务数据按年份升序排列，取最近 10 年且落在 K 线范围内的
            fin_df = fin_df.sort_values("报告期", ascending=False)
            historical_pes = []
            for _, row in fin_df.iterrows():
                try:
                    # 报告期是纯年份（如 2024），转为当年最后一天
                    year_str = str(row["报告期"]).strip()
                    year = int(year_str[:4])
                    report_date = pd.Timestamp(year=year, month=12, day=31)
                    # 只匹配有价格数据的年份
                    if report_date < price_df["date"].min():
                        continue

                    eps_str = row["基本每股收益"]
                    if eps_str is None or str(eps_str).lower() in ("false", "true", ""):
                        continue
                    eps = float(eps_str)
                    if eps <= 0:
                        continue

                    nearby = price_df[price_df["date"] <= report_date]
                    if nearby.empty:
                        continue
                    close_price = float(nearby.iloc[-1]["close"])
                    if close_price > 0:
                        historical_pes.append(round(close_price / eps, 2))
                except (ValueError, TypeError, KeyError):
                    continue

            if len(historical_pes) < 3:
                return {
                    "pe_percentile_pct": None,
                    "pe_percentile_note": f"仅{len(historical_pes)}年有效PE数据，需≥3年",
                }

            count_below = sum(1 for pe in historical_pes if pe < current_pe)
            percentile = round(count_below / len(historical_pes) * 100, 1)

            return {
                "pe_percentile_pct":   percentile,
                "pe_percentile_years": len(historical_pes),
                "pe_percentile_lo":    round(min(historical_pes), 1),
                "pe_percentile_hi":    round(max(historical_pes), 1),
                "pe_history":          sorted(historical_pes),
            }
        except Exception as e:
            return {"pe_percentile_pct": None, "pe_percentile_note": f"计算失败: {e}"}

    # ── READ handlers ─────────────────────────

    def _tool_get_stock_data(self, stock_code: str) -> dict:
        # 腾讯行情（Docker 内可用，延迟低）
        try:
            quote = fetch_tencent_quote(stock_code)
        except Exception as e:
            return {"error": f"获取行情失败: {e}", "stock_code": stock_code}

        price = quote["price"]
        name = quote["name"]
        pe_raw = quote["pe_ttm"]

        # TTM 股息率
        ttm_yield = None
        try:
            import akshare as ak
            import pandas as pd
            div_df = ak.stock_history_dividend_detail(
                symbol=stock_code, indicator="分红"
            )
            if not div_df.empty and price > 0:
                div_df["除权除息日"] = pd.to_datetime(
                    div_df["除权除息日"], errors="coerce"
                )
                cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
                recent = div_df[div_df["除权除息日"] >= cutoff]
                if not recent.empty:
                    total_div_per_10 = recent["派息"].astype(float).sum()
                    div_per_share = total_div_per_10 / 10
                    ttm_yield = round(div_per_share / price * 100, 2)
        except Exception:
            pass

        # PE 历史百分位
        pe_pct_info = {}
        try:
            pe_num = float(pe_raw) if pe_raw else None
        except (ValueError, TypeError):
            pe_num = None

        if pe_num and pe_num > 0:
            pe_pct_info = self._compute_pe_percentile(stock_code, pe_num)
        else:
            pe_pct_info = {"pe_percentile_pct": None, "pe_percentile_note": "当前PE无效"}

        return {
            "stock_code":    stock_code,
            "name":          name,
            "price":         price,
            "pe_ttm":        pe_raw,
            "pb":            quote["pb"],
            "market_cap_bn": quote["market_cap_bn"],
            "ttm_yield_pct": ttm_yield,
            "52w_high":      quote["52w_high"],
            "52w_low":       quote["52w_low"],
            **pe_pct_info,
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
                div_per_share = float(row["派息"]) / 10
            except (ValueError, TypeError):
                div_per_share = None

            records.append({
                "date":            str(row["除权除息日"].date()),
                "div_per_share":   div_per_share,
                "bonus_per_10":    row.get("送股"),
                "transfer_per_10": row.get("转增"),
            })

        return records

    def _tool_get_financials(self, stock_code: str) -> list:
        import akshare as ak

        df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按年度")
        if df.empty:
            return []

        df = df.head(5)

        keep_cols = [
            "报告期", "营业总收入", "净利润", "基本每股收益",
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
