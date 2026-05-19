"""
tools.py -- 工具定义 + 执行器

READ  tools (9) : Agent 可自主调用，无需请示
WRITE tools (7) : 必须用户明确指令 + 展示内容 + 等待"确认"后才调用
"""

import json
import os
import re
import sqlite3
import threading
from datetime import date, datetime

from memory import DB_PATH, MemoryManager, _enable_wal_safely
from metrics import console_timer, get_tracer

PE_PERCENTILE_YEARS = 3


# ─────────────────────────────────────────────
# Tool Schema（传给 LLM 的格式）
# ─────────────────────────────────────────────

READ_TOOLS = [
    {
        "name": "calculate_dividend_reinvestment",
        "description": (
            "计算A股按月定投与股息复投的长期估算：每月买入多少手、"
            "持有多少年、股息是否复投，返回累计股息、第N年当年股息、"
            "复投新增股数和剩余现金。适合定投/股息复投/长期收益问题。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "6位股票代码，如 601398"},
                "monthly_lots": {"type": "integer", "description": "每月定投手数，1手=100股"},
                "years": {"type": "integer", "description": "测算年限"},
                "dividend_reinvest": {"type": "boolean", "description": "是否用股息复投"},
            },
            "required": ["stock_code", "monthly_lots", "years"],
        },
    },
    {
        "name": "get_stock_data",
        "description": "获取A股实时行情与估值：当前价格、PE、PB、TTM股息率、PE历史百分位（当前PE在近3年中的分位，越低越便宜）。Agent可自主调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "6位股票代码，如 600028"},
                "include_pe_percentile": {
                    "type": "boolean",
                    "description": "是否同步计算PE历史百分位；默认false以避免AKShare慢接口导致超时。用户明确要求历史分位时再设为true。",
                },
                "async_pe_percentile": {
                    "type": "boolean",
                    "description": "include_pe_percentile=false且缓存未命中时，是否后台预热PE历史百分位缓存；默认true。",
                },
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
                "alert_price_below": {
                    "type": "number",
                    "description": "强提醒价格阈值，现价低于或等于该价格时提醒",
                },
                "watch_price_below": {
                    "type": "number",
                    "description": "观察价格阈值，现价低于或等于该价格时开始留意",
                },
                "alert_note": {
                    "type": "string",
                    "description": "基本面、财报或其他人工复核提醒，例如等待年报落地后关注ROE和营收增速",
                },
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
WRITE_TOOL_NAMES = {tool["name"] for tool in WRITE_TOOLS}


STOCK_NAME_ALIASES = {
    "工商银行": "601398",
    "工行": "601398",
    "格力电器": "000651",
    "格力": "000651",
    "伊利股份": "600887",
    "伊利": "600887",
}


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
    # 腾讯行情字段覆盖价格、PE/PB、市值和52周区间，适合做低延迟第一手快照；
    # 更重的财务、分红数据再由 AKShare 补齐。
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


def resolve_stock_code(text: str) -> str | None:
    # 先识别显式6位代码，再走少量高频别名。
    # 业务上宁可少猜，也不要把模糊股票名误映射到错误标的。
    match = re.search(r"(?<!\d)\d{6}(?!\d)", text or "")
    if match:
        return match.group(0)
    for name, code in STOCK_NAME_ALIASES.items():
        if name in (text or ""):
            return code
    return None


def calculate_dividend_reinvestment_projection(
    *,
    stock_code: str,
    stock_name: str,
    price: float,
    annual_dividend_per_share: float,
    monthly_lots: int,
    years: int,
    dividend_reinvest: bool = True,
    dividend_source_date: str | None = None,
) -> dict:
    # 这是一个静态口径的长期现金流模型：
    # 假设价格和每股分红不变，定投按月买入，分红按年结算，并按A股整手复投。
    if price <= 0:
        return {"error": "当前价格无效，无法计算复投股数", "stock_code": stock_code}
    if annual_dividend_per_share <= 0:
        return {"error": "每股分红无效，无法计算股息收益", "stock_code": stock_code}
    if monthly_lots <= 0:
        return {"error": "每月定投手数必须大于0", "stock_code": stock_code}
    if years <= 0:
        return {"error": "测算年限必须大于0", "stock_code": stock_code}

    monthly_shares = monthly_lots * 100
    annual_buy_shares = monthly_shares * 12
    shares = 0
    reinvested_shares = 0
    carry_cash = 0.0
    cumulative_dividend = 0.0
    yearly = []

    for year in range(1, years + 1):
        starting_shares = shares
        # 每年先累计12个月定投股数，再按年末持股计算当年现金分红。
        shares += annual_buy_shares
        cash_dividend = round(shares * annual_dividend_per_share, 2)
        cumulative_dividend = round(cumulative_dividend + cash_dividend, 2)

        reinvest_shares = 0
        available_cash = carry_cash + cash_dividend
        if dividend_reinvest:
            # A股买入以100股为一手，股息不够一手时留作下年现金结转。
            lot_cost = price * 100
            reinvest_lots = int(available_cash // lot_cost)
            reinvest_shares = reinvest_lots * 100
            if reinvest_shares:
                shares += reinvest_shares
                reinvested_shares += reinvest_shares
            carry_cash = round(available_cash - reinvest_shares * price, 2)
        else:
            carry_cash = round(available_cash, 2)

        yearly.append({
            "year": year,
            "starting_shares": starting_shares,
            "regular_buy_shares": annual_buy_shares,
            "cash_dividend": cash_dividend,
            "reinvested_shares": reinvest_shares,
            "ending_shares": shares,
            "carry_cash": carry_cash,
        })

    last_year = yearly[-1]
    return {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "price": round(price, 3),
        "annual_dividend_per_share": round(annual_dividend_per_share, 4),
        "dividend_source_date": dividend_source_date,
        "monthly_lots": monthly_lots,
        "monthly_shares": monthly_shares,
        "years": years,
        "dividend_reinvest": dividend_reinvest,
        "regular_buy_shares": annual_buy_shares * years,
        "reinvested_shares": reinvested_shares,
        "ending_shares": shares,
        "cumulative_dividend": cumulative_dividend,
        "last_year_dividend": last_year["cash_dividend"],
        "remaining_cash": carry_cash,
        "assumption": "当前股价和最近年度每股现金分红长期不变；按年末分红并按A股整手复投，剩余现金结转。",
        "yearly": yearly,
    }


class ToolExecutor:
    _pe_cache_lock = threading.Lock()
    _pe_cache_warming: set[str] = set()
    _pe_cache_warming_lock = threading.Lock()

    def __init__(self, memory: MemoryManager):
        self.memory = memory
        self._init_pe_cache()

    def execute(self, tool_name: str, tool_input: dict,
                allow_write: bool = False) -> str:
        with get_tracer().tool_call(tool_name, tool_input) as rec:
            with console_timer("工具执行", f"{tool_name} input={tool_input}"):
                try:
                    if tool_name in WRITE_TOOL_NAMES and not allow_write:
                        # 工具层再次兜底：即使 LLM 或调用方绕过 Agent 流程，
                        # 写入数据库也必须显式带 allow_write=True。
                        out = json.dumps({
                            "error": "写工具需要用户确认后才能执行",
                            "tool": tool_name,
                            "requires_confirmation": True,
                        }, ensure_ascii=False)
                        rec.error = "requires_confirmation"
                        rec.result_chars = len(out)
                        return out

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
    def _pe_cache_db_path() -> str:
        return os.environ.get("DB_PATH", DB_PATH)

    @classmethod
    def _init_pe_cache(cls):
        db_path = cls._pe_cache_db_path()
        try:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            with cls._pe_cache_lock:
                conn = sqlite3.connect(db_path, timeout=30)
                try:
                    _enable_wal_safely(conn, "pe_percentile_cache")
                    # PE历史百分位的原始历史PE按交易日缓存。
                    # 当天多次问同一只股票时，只用当前PE重新算分位，避免反复拉远端数据。
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS pe_percentile_cache (
                            stock_code           TEXT PRIMARY KEY,
                            trade_date           TEXT NOT NULL,
                            pe_history           TEXT NOT NULL,
                            pe_percentile_years  INTEGER NOT NULL,
                            pe_percentile_lo     REAL NOT NULL,
                            pe_percentile_hi     REAL NOT NULL,
                            updated_at           TEXT NOT NULL
                        )
                    """)
                    conn.commit()
                finally:
                    conn.close()
        except sqlite3.Error as e:
            print(f"  [PE缓存] 初始化失败，降级为无缓存模式: {e}", flush=True)
        except OSError as e:
            print(f"  [PE缓存] 初始化目录失败，降级为无缓存模式: {e}", flush=True)

    @staticmethod
    def _percentile_from_history(current_pe: float, historical_pes: list[float]) -> dict:
        # 百分位含义：历史PE中有多少比例低于当前PE。
        # 数值越低，说明当前估值越接近历史低位。
        count_below = sum(1 for pe in historical_pes if pe < current_pe)
        percentile = round(count_below / len(historical_pes) * 100, 1)
        return {
            "pe_percentile_pct":   percentile,
            "pe_percentile_years": len(historical_pes),
            "pe_percentile_lo":    round(min(historical_pes), 1),
            "pe_percentile_hi":    round(max(historical_pes), 1),
            "pe_history":          sorted(historical_pes),
        }

    @classmethod
    def _load_cached_pe_percentile(cls, stock_code: str, current_pe: float) -> dict | None:
        today = date.today().isoformat()
        try:
            with cls._pe_cache_lock:
                conn = sqlite3.connect(cls._pe_cache_db_path(), timeout=30)
                conn.row_factory = sqlite3.Row
                try:
                    row = conn.execute(
                        """
                        SELECT pe_history, pe_percentile_years FROM pe_percentile_cache
                        WHERE stock_code = ? AND trade_date = ?
                        """,
                        (stock_code, today),
                    ).fetchone()
                finally:
                    conn.close()
        except sqlite3.Error as e:
            print(f"  [PE缓存] 读取失败，跳过缓存: {e}", flush=True)
            return None

        if row is None:
            return None
        if row["pe_percentile_years"] != PE_PERCENTILE_YEARS:
            return None

        try:
            historical_pes = json.loads(row["pe_history"])
            if len(historical_pes) < 3:
                return None
            result = cls._percentile_from_history(current_pe, historical_pes)
            result["pe_percentile_cached"] = True
            return result
        except Exception:
            return None

    @classmethod
    def _save_cached_pe_history(cls, stock_code: str, pe_info: dict):
        historical_pes = pe_info.get("pe_history")
        if not historical_pes or len(historical_pes) < 3:
            return

        now = datetime.now().isoformat(timespec="seconds")
        today = date.today().isoformat()
        try:
            with cls._pe_cache_lock:
                conn = sqlite3.connect(cls._pe_cache_db_path(), timeout=30)
                try:
                    conn.execute(
                        """
                        INSERT INTO pe_percentile_cache (
                            stock_code, trade_date, pe_history, pe_percentile_years,
                            pe_percentile_lo, pe_percentile_hi, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(stock_code) DO UPDATE SET
                            trade_date = excluded.trade_date,
                            pe_history = excluded.pe_history,
                            pe_percentile_years = excluded.pe_percentile_years,
                            pe_percentile_lo = excluded.pe_percentile_lo,
                            pe_percentile_hi = excluded.pe_percentile_hi,
                            updated_at = excluded.updated_at
                        """,
                        (
                            stock_code,
                            today,
                            json.dumps(sorted(historical_pes), ensure_ascii=False),
                            pe_info.get("pe_percentile_years") or len(historical_pes),
                            pe_info.get("pe_percentile_lo") or min(historical_pes),
                            pe_info.get("pe_percentile_hi") or max(historical_pes),
                            now,
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except sqlite3.Error as e:
            print(f"  [PE缓存] 写入失败，跳过缓存: {e}", flush=True)

    @staticmethod
    def _compact_pe_percentile(pe_info: dict) -> dict:
        compact = {k: v for k, v in pe_info.items() if k != "pe_history"}
        if "pe_history" in pe_info:
            compact["pe_percentile_range"] = {
                "low": pe_info.get("pe_percentile_lo"),
                "high": pe_info.get("pe_percentile_hi"),
                "years": pe_info.get("pe_percentile_years"),
            }
        return compact

    @classmethod
    def _warm_pe_percentile_cache_async(cls, stock_code: str, current_pe: float):
        with cls._pe_cache_warming_lock:
            if stock_code in cls._pe_cache_warming:
                return
            cls._pe_cache_warming.add(stock_code)

        def worker():
            try:
                with console_timer("后台计算", f"PE历史百分位 {stock_code}"):
                    pe_info = cls._compute_pe_percentile(stock_code, current_pe)
                cls._save_cached_pe_history(stock_code, pe_info)
            finally:
                with cls._pe_cache_warming_lock:
                    cls._pe_cache_warming.discard(stock_code)

        thread = threading.Thread(
            target=worker,
            name=f"pe-percentile-{stock_code}",
            daemon=True,
        )
        thread.start()

    @staticmethod
    def _compute_pe_percentile(stock_code: str, current_pe: float,
                               fin_df=None, price_df=None) -> dict:
        """计算当前 PE 在近 3 年历史中的分位。返回 pe_percentile_pct 等字段。

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

            # 3年日K线（腾讯源，Docker 内可用）
            if price_df is None:
                import akshare as ak
                end_date = pd.Timestamp.now().strftime("%Y%m%d")
                start_date = (
                    pd.Timestamp.now() - pd.DateOffset(years=PE_PERCENTILE_YEARS)
                ).strftime("%Y%m%d")
                prefix = exchange_prefix(stock_code)
                price_df = ak.stock_zh_a_hist_tx(
                    symbol=f"{prefix}{stock_code}", start_date=start_date,
                    end_date=end_date, adjust="qfq",
                )
            if price_df.empty:
                return {"pe_percentile_pct": None, "pe_percentile_note": "无历史价格数据"}

            price_df["date"] = pd.to_datetime(price_df["date"])
            min_price_date = price_df["date"].min()

            # 每年报告期找最近交易日收盘价，计算该年 PE
            # 只取最近 3 年且落在 K 线范围内的年度财务数据。
            fin_df = fin_df.sort_values("报告期", ascending=False)
            historical_pes = []
            for _, row in fin_df.iterrows():
                try:
                    # 报告期是纯年份（如 2024），转为当年最后一天
                    year_str = str(row["报告期"]).strip()
                    year = int(year_str[:4])
                    report_date = pd.Timestamp(year=year, month=12, day=31)
                    # 只匹配有价格数据的年份
                    if report_date < min_price_date:
                        continue

                    eps_str = row["基本每股收益"]
                    if eps_str is None or str(eps_str).lower() in ("false", "true", ""):
                        continue
                    eps = float(eps_str)
                    if eps <= 0:
                        # EPS为负时PE失真，不能纳入“估值便宜/贵”的历史分位样本。
                        continue

                    nearby = price_df[price_df["date"] <= report_date]
                    if nearby.empty:
                        continue
                    close_price = float(nearby.iloc[-1]["close"])
                    if close_price > 0:
                        historical_pes.append(round(close_price / eps, 2))
                    if len(historical_pes) >= PE_PERCENTILE_YEARS:
                        break
                except (ValueError, TypeError, KeyError):
                    continue

            if len(historical_pes) < 3:
                return {
                    "pe_percentile_pct": None,
                    "pe_percentile_note": f"仅{len(historical_pes)}年有效PE数据，需≥3年",
                }

            return ToolExecutor._percentile_from_history(current_pe, historical_pes)
        except Exception as e:
            return {"pe_percentile_pct": None, "pe_percentile_note": f"计算失败: {e}"}

    # ── READ handlers ─────────────────────────

    def _tool_calculate_dividend_reinvestment(
        self,
        stock_code: str,
        monthly_lots: int,
        years: int,
        dividend_reinvest: bool = True,
    ) -> dict:
        import akshare as ak
        import pandas as pd

        try:
            with console_timer("数据查询", f"腾讯行情 {stock_code}"):
                quote = fetch_tencent_quote(stock_code)
        except Exception as e:
            return {"error": f"获取行情失败: {e}", "stock_code": stock_code}

        price = quote.get("price") or 0.0
        if price <= 0:
            return {"error": "当前价格无效，无法计算复投股数", "stock_code": stock_code}

        with console_timer("数据查询", f"AKShare 最近分红 {stock_code}"):
            div_df = ak.stock_history_dividend_detail(symbol=stock_code, indicator="分红")
        if div_df.empty:
            return {"error": "分红数据不可用，无法计算股息复投", "stock_code": stock_code}

        div_df["除权除息日"] = pd.to_datetime(div_df["除权除息日"], errors="coerce")
        div_df = div_df.dropna(subset=["除权除息日"]).sort_values("除权除息日", ascending=False)
        div_df = div_df[div_df["除权除息日"] <= pd.Timestamp.now()]
        latest = None
        for _, row in div_df.iterrows():
            try:
                div_per_share = float(row["派息"]) / 10
            except (ValueError, TypeError, KeyError):
                continue
            if div_per_share > 0:
                # 取最近一次有效现金分红作为静态年化口径，避免送转股记录污染现金分红测算。
                latest = (row, div_per_share)
                break

        if latest is None:
            return {"error": "未找到有效现金分红数据，无法计算股息复投", "stock_code": stock_code}

        row, annual_dividend_per_share = latest
        return calculate_dividend_reinvestment_projection(
            stock_code=stock_code,
            stock_name=quote.get("name") or stock_code,
            price=float(price),
            annual_dividend_per_share=annual_dividend_per_share,
            monthly_lots=int(monthly_lots),
            years=int(years),
            dividend_reinvest=bool(dividend_reinvest),
            dividend_source_date=str(row["除权除息日"].date()),
        )

    def _tool_get_stock_data(
        self,
        stock_code: str,
        include_pe_percentile: bool = False,
        async_pe_percentile: bool = True,
    ) -> dict:
        # 腾讯行情（Docker 内可用，延迟低）
        try:
            with console_timer("数据查询", f"腾讯行情 {stock_code}"):
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
            with console_timer("数据查询", f"AKShare 分红 {stock_code}"):
                div_df = ak.stock_history_dividend_detail(
                    symbol=stock_code, indicator="分红"
                )
            if not div_df.empty and price > 0:
                div_df["除权除息日"] = pd.to_datetime(
                    div_df["除权除息日"], errors="coerce"
                )
                cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
                # TTM股息率只看最近12个月已除权的现金派息，更贴近“当前买入能对应的股息水平”。
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
            with console_timer("数据计算", f"PE历史百分位缓存 {stock_code}"):
                pe_pct_info = self._load_cached_pe_percentile(stock_code, pe_num)
            if pe_pct_info is None and include_pe_percentile:
                # 显式要求时才同步拉年度EPS和K线；这是全项目最重的行情计算之一。
                with console_timer("数据计算", f"PE历史百分位 {stock_code}"):
                    pe_pct_info = self._compute_pe_percentile(stock_code, pe_num)
                self._save_cached_pe_history(stock_code, pe_pct_info)
            elif pe_pct_info is None:
                if async_pe_percentile:
                    self._warm_pe_percentile_cache_async(stock_code, pe_num)
                pe_pct_info = {
                    "pe_percentile_pct": None,
                    "pe_percentile_note": "未同步计算；后台缓存中" if async_pe_percentile else "未同步计算",
                }
        else:
            pe_pct_info = {"pe_percentile_pct": None, "pe_percentile_note": "当前PE无效"}
        pe_pct_info = self._compact_pe_percentile(pe_pct_info)

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

        with console_timer("数据查询", f"AKShare 分红历史 {stock_code}"):
            df = ak.stock_history_dividend_detail(symbol=stock_code, indicator="分红")
        if df.empty:
            return []

        df["除权除息日"] = pd.to_datetime(df["除权除息日"], errors="coerce")
        df = df.dropna(subset=["除权除息日"])

        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        # 分红历史保留逐次记录，而不是提前汇总，方便 Agent 判断稳定性和间断年份。
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

        with console_timer("数据查询", f"AKShare 财务摘要 {stock_code}"):
            df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按年度")
        if df.empty:
            return []

        df = df.head(5)

        # 财务摘要只取投资判断最常用的质量指标：收入、利润、EPS、ROE、净资产和负债率。
        # 更细的财报科目不进工具结果，避免挤占 LLM 上下文。
        keep_cols = [
            "报告期", "营业总收入", "净利润", "基本每股收益",
            "净资产收益率", "每股净资产", "资产负债率",
        ]
        existing = [c for c in keep_cols if c in df.columns]
        df = df[existing]

        return df.to_dict(orient="records")

    def _tool_get_ah_premium(self, stock_code: str) -> dict:
        import akshare as ak

        with console_timer("数据查询", "AKShare AH溢价"):
            df = ak.stock_zh_ah_spot_em()
        if df.empty:
            return {"error": "AH 数据获取失败"}

        code_col = next((c for c in df.columns if "代码" in c and "A" in c), None)
        if code_col is None:
            code_col = df.columns[0]

        row = df[df[code_col] == stock_code]
        if row.empty:
            # 不是所有A股都有H股映射；返回 note 让 Agent 明确说明“该维度不适用”。
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
        with console_timer("数据查询", "SQLite decisions"):
            return self.memory.decisions.search_decisions(
                stock_code=stock_code, keyword=keyword, limit=limit
            )

    def _tool_get_positions(self) -> list:
        with console_timer("数据查询", "SQLite positions"):
            return self.memory.decisions.get_positions()

    def _tool_get_watchlist(self) -> list:
        with console_timer("数据查询", "SQLite watchlist"):
            return self.memory.decisions.get_watchlist()

    def _tool_search_retrospectives(self, decision_id: int) -> list:
        with console_timer("数据查询", "SQLite retrospectives"):
            return self.memory.decisions.search_retrospectives(decision_id)

    def _tool_retrieve_memory(self, query: str, top_k: int = 4) -> list:
        with console_timer("数据查询", "记忆混合检索"):
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
