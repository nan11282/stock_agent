"""
scheduler.py -- 定时扫描 + 邮件

每天15:30收盘后自动扫描持仓/自选/市场发现，生成报告并发邮件。
支持手动触发：python scheduler.py --now
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import schedule
import time

from adapters import Message, ToolResult
from memory import MemoryManager
from mailer import send_report
from tools import fetch_tencent_quote, READ_TOOLS, ToolExecutor
from telegram_bot import start_bot
from runtime import build_default_llm
from metrics import console_timer


def _read_tool_max_workers(n: int) -> int:
    return min(max(1, int(os.environ.get("READ_TOOL_MAX_WORKERS", "5"))), n)


class DailyScanner:
    def __init__(self):
        self.memory = MemoryManager()
        self.llm = build_default_llm()

    # ── 第一层：持仓扫描 ─────────────────────

    def scan_positions(self) -> list[dict]:
        import akshare as ak

        # 持仓扫描回答的是“我已经买了的东西今天是否偏离预期”。
        # 因此核心信号是成本价相对现价的盈亏幅度，再补充 PE 做估值观察。
        positions = self.memory.decisions.get_positions()
        results = []

        for pos in positions:
            code = pos["stock_code"]
            try:
                quote = fetch_tencent_quote(code)
                price = quote["price"]
                pe = quote["pe_ttm"]

                cost = pos["cost_price"]
                pnl_pct = round((price - cost) / cost * 100, 2) if cost else None

                signal = "normal"
                summary_parts = [f"现价{price}"]
                if pnl_pct is not None:
                    summary_parts.append(f"盈亏{pnl_pct}%")
                    if pnl_pct < -10:
                        # 跌破成本较多时先提醒复核基本面/仓位，而不是自动给卖出建议。
                        signal = "alert"
                        summary_parts.append("跌幅超10%需关注")
                    elif pnl_pct > 30:
                        # 大幅浮盈触发止盈检查，适合高股息/低估值策略里的再平衡提醒。
                        signal = "alert"
                        summary_parts.append("涨幅超30%可考虑止盈")

                if pe:
                    summary_parts.append(f"PE={pe}")

                results.append({
                    "scope": "positions",
                    "stock_code": code,
                    "stock_name": pos["stock_name"],
                    "signal": signal,
                    "summary": " | ".join(summary_parts),
                    "price": price,
                    "cost": cost,
                    "pnl_pct": pnl_pct,
                })
            except Exception as e:
                results.append({
                    "scope": "positions",
                    "stock_code": code,
                    "stock_name": pos["stock_name"],
                    "signal": "alert",
                    "summary": f"数据获取失败: {e}",
                })

        return results

    # ── 第二层：自选股监控 ────────────────────

    def scan_watchlist(self) -> list[dict]:
        import akshare as ak
        import pandas as pd

        watchlist = self.memory.decisions.get_watchlist()
        results = []

        for w in watchlist:
            code = w["stock_code"]
            try:
                quote = fetch_tencent_quote(code)
                price = quote["price"]

                # 计算 TTM 股息率
                ttm_yield = None
                try:
                    div_df = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
                    if not div_df.empty and price > 0:
                        div_df["除权除息日"] = pd.to_datetime(div_df["除权除息日"], errors="coerce")
                        cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
                        recent = div_df[div_df["除权除息日"] >= cutoff]
                        if not recent.empty:
                            total = recent["派息"].astype(float).sum()
                            ttm_yield = round(total / 10 / price * 100, 2)
                except Exception:
                    pass

                signal = "normal"
                summary_parts = [f"现价{price}"]

                if ttm_yield is not None:
                    summary_parts.append(f"TTM股息率{ttm_yield}%")
                    if w.get("alert_yield") and ttm_yield >= w["alert_yield"]:
                        # 自选股的提醒不是“买入指令”，只是说明用户预设的观察条件已经触发。
                        signal = "alert"
                        summary_parts.append(f"达到股息率阈值{w['alert_yield']}%")

                results.append({
                    "scope": "watchlist",
                    "stock_code": code,
                    "stock_name": w["stock_name"],
                    "signal": signal,
                    "summary": " | ".join(summary_parts),
                })
            except Exception as e:
                results.append({
                    "scope": "watchlist",
                    "stock_code": code,
                    "stock_name": w["stock_name"],
                    "signal": "alert",
                    "summary": f"数据获取失败: {e}",
                })

        return results

    # ── 第三层：市场发现 ──────────────────────

    def scan_discovery(self) -> list[dict]:
        import akshare as ak

        results = []

        # 已知持仓和自选的代码，用于排除
        # 市场发现优先补充新机会，避免日报反复推荐用户已经跟踪的标的。
        known_codes = set()
        for p in self.memory.decisions.get_positions():
            known_codes.add(p["stock_code"])
        for w in self.memory.decisions.get_watchlist():
            known_codes.add(w["stock_code"])

        # ── 维度1：AH折价（溢价率<80的，即A股比H股便宜）──
        # AH折价是这里的量化发现入口：同一公司两地价格差过大时，
        # 说明A股可能出现相对低估，需要进一步看流动性、汇率和业务基本面。
        try:
            ah_df = ak.stock_zh_ah_spot_em()
            if not ah_df.empty:
                premium_col = next(
                    (c for c in ah_df.columns if "溢价" in c), None
                )
                name_col = next(
                    (c for c in ah_df.columns if "名称" in c), None
                )
                code_col = next(
                    (c for c in ah_df.columns if "代码" in c and "A" in c),
                    ah_df.columns[0],
                )
                if premium_col:
                    ah_df[premium_col] = ah_df[premium_col].astype(float, errors="ignore")
                    cheap = ah_df[ah_df[premium_col] < 80].head(5)
                    for _, row in cheap.iterrows():
                        code = str(row[code_col])
                        name = row[name_col] if name_col else code
                        results.append({
                            "scope": "discovery",
                            "stock_code": code,
                            "stock_name": str(name),
                            "signal": "opportunity",
                            "summary": f"AH溢价率{row[premium_col]}%，A股折价",
                        })
        except Exception:
            pass

        # ── 维度2/3：低估板块和高息发现（简化为提示信息）──
        try:
            results.append({
                "scope": "discovery",
                "stock_code": "",
                "stock_name": "板块扫描",
                "signal": "normal",
                "summary": "银行/煤炭/电力/交通基础设施板块待人工深入分析",
            })
        except Exception:
            pass

        return results

    # ── LLM 深度分析（ReAct + 读工具）─────────

    ANALYSIS_PROMPT = """你是 Coco 的A股投资分析师。今天是 {today}，已收盘。

【任务】筛选 1-2 只当前最值得关注或操作的股票，深入分析并给出明确建议。

【分析维度——每只推荐股票必须覆盖】
1. 当前估值：PE、PB、PE历史百分位（越低越便宜）、TTM股息率，与自身历史区间对比
2. 公司业务：做什么的，行业地位，护城河/竞争优势
3. 财务健康：近5年ROE趋势、营收/利润增长、资产负债率、现金流状况
4. 分红记录：是否持续分红、股息率稳定性
5. 催化剂/风险：近期可能的驱动事件、需要警惕的因素
6. 操作建议：买入/观望/卖出，什么价位区间合理，仓位建议

【风格】
- 先调工具拿数据，再分析，绝不要凭空编造
- 结论要明确，不要"可能""或许""值得关注"这种废话
- 如果某只股票数据不全，诚实说明哪些维度缺失
- 总字数 800-1500 字"""

    def _deep_analysis(self, pos_results: list, watch_results: list,
                       disc_results: list) -> str:
        executor = ToolExecutor(self.memory)

        # 将扫描结果整理成上下文
        # 规则扫描只负责“发现异常/机会线索”，最终日报观点交给 LLM 再调用读工具补证据。
        all_data = json.dumps({
            "持仓扫描": pos_results,
            "自选监控": watch_results,
            "市场发现": disc_results,
        }, ensure_ascii=False, indent=2)

        system = self.ANALYSIS_PROMPT.format(today=datetime.now().strftime("%Y-%m-%d"))
        messages = [Message(role="user", text=(
            f"今日A股扫描数据如下，请先用工具拉取关键数据，"
            f"再从中挑选1-2只最有价值的股票做深度分析：\n\n{all_data}"
        ))]

        final_text = ""
        for step in range(8):
            with console_timer("scheduler", f"LLM step={step + 1}"):
                resp = self.llm.chat(messages, READ_TOOLS, system)
            if not resp.tool_calls:
                final_text = resp.text or ""
                break

            def run_tool(tc):
                # 日报里的深度分析同样只开放读工具，避免定时任务在无人确认时写入决策库。
                print(f"  [分析工具] {tc.name} {tc.input}")
                with console_timer("scheduler", f"tool {tc.name}"):
                    return executor.execute(tc.name, tc.input)

            if len(resp.tool_calls) <= 1:
                results = [run_tool(tc) for tc in resp.tool_calls]
            else:
                workers = _read_tool_max_workers(len(resp.tool_calls))
                with console_timer("scheduler", f"parallel read tools n={len(resp.tool_calls)} workers={workers}"):
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        results = list(pool.map(run_tool, resp.tool_calls))

            messages.append(Message(role="assistant",
                text=resp.text,
                tool_calls=list(resp.tool_calls),
                reasoning_content=resp.reasoning_content,
            ))
            messages.append(Message(role="user", tool_results=[
                ToolResult(tool_call_id=tc.id, content=r)
                for tc, r in zip(resp.tool_calls, results)
            ]))
        else:
            final_text = f"[警告] 分析达到最大步数 {8}，强制终止。\n\n{resp.text or ''}"

        return final_text or "分析生成失败"

    # ── 主入口 ────────────────────────────────

    def run(self):
        print(f"[{datetime.now().isoformat()}] 开始每日扫描...")

        # 三层扫描分别覆盖：已有持仓风险、自选触发条件、组合外机会。
        # 先做结构化扫描，再用 LLM 生成可读投资报告。
        with console_timer("每日扫描", "持仓扫描"):
            pos_results = self.scan_positions()
        with console_timer("每日扫描", "自选监控"):
            watch_results = self.scan_watchlist()
        with console_timer("每日扫描", "市场发现"):
            disc_results = self.scan_discovery()

        print("  启动深度分析引擎（LLM + 读工具）...")
        with console_timer("每日扫描", "LLM深度分析"):
            analysis = self._deep_analysis(pos_results, watch_results, disc_results)
        print(f"  分析完成: {analysis[:120]}...")

        # 发邮件
        try:
            with console_timer("每日扫描", "邮件发送"):
                send_report(
                    summary=analysis,
                    positions=pos_results,
                    watchlist=watch_results,
                    discovery=disc_results,
                )
            print("  邮件发送成功")
        except Exception as e:
            print(f"  邮件发送失败: {e}")

        # 结果写入 scan_results 表
        # 保存扫描原始信号，便于后续追踪“某个提醒持续出现了几天”。
        with console_timer("每日扫描", "scan_results写入"):
            for item in pos_results + watch_results + disc_results:
                self.memory.decisions.save_scan_result(item)

        print(f"[{datetime.now().isoformat()}] 扫描完成")


def _already_scanned_today(scanner: "DailyScanner") -> bool:
    # 容器重启后用于防止同一天重复补发日报。
    today = datetime.now().strftime("%Y-%m-%d")
    row = scanner.memory.decisions.conn.execute(
        "SELECT 1 FROM scan_results WHERE scanned_at >= ? LIMIT 1", (today,)
    ).fetchone()
    return row is not None


def main():
    parser = argparse.ArgumentParser(description="A股投资助理 - 定时扫描")
    parser.add_argument("--now", action="store_true", help="立即执行一次扫描")
    args = parser.parse_args()

    scanner = DailyScanner()

    if args.now:
        # 手动触发用于调试或盘后临时补发，不受当天是否已扫描限制。
        scanner.run()
        return

    # 启动时补发：若已过 15:30 且今天尚未扫描，立即执行一次
    # 这解决容器在收盘后才启动时错过 schedule 定时点的问题。
    now = datetime.now()
    trigger = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now >= trigger and not _already_scanned_today(scanner):
        print(f"[{now.isoformat()}] 检测到今日扫描未执行，立即补发...")
        scanner.run()

    print("定时任务已启动，每天 15:30 执行扫描...")
    schedule.every().day.at("15:30").do(scanner.run)

    # schedule 循环放到守护线程，主线程留给 Telegram bot
    def _schedule_loop():
        while True:
            schedule.run_pending()
            time.sleep(60)

    import threading
    threading.Thread(target=_schedule_loop, daemon=True).start()

    # Telegram 聊天助理（阻塞主线程，事件驱动长轮询）
    start_bot()


if __name__ == "__main__":
    main()
