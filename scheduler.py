"""
scheduler.py -- 定时扫描 + 邮件

每天15:30收盘后自动扫描持仓/自选/市场发现，生成报告并发邮件。
支持手动触发：python scheduler.py --now
"""

import argparse
import json
from datetime import datetime

import schedule
import time

import os

from adapters import OpenAIAdapter, Message
from memory import MemoryManager
from mailer import send_report


class DailyScanner:
    def __init__(self):
        self.memory = MemoryManager()
        self.llm = OpenAIAdapter(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY_stock_agent"),
        )

    # ── 第一层：持仓扫描 ─────────────────────

    def scan_positions(self) -> list[dict]:
        import akshare as ak
        import pandas as pd

        positions = self.memory.decisions.get_positions()
        results = []

        for pos in positions:
            code = pos["stock_code"]
            try:
                info_df = ak.stock_individual_info_em(symbol=code)
                info = dict(zip(info_df["item"], info_df["value"]))
                price = float(info.get("最新", 0) or 0)
                pe = info.get("市盈率(动态)")

                cost = pos["cost_price"]
                pnl_pct = round((price - cost) / cost * 100, 2) if cost else None

                # 判断异常
                signal = "normal"
                summary_parts = [f"现价{price}"]
                if pnl_pct is not None:
                    summary_parts.append(f"盈亏{pnl_pct}%")
                    if pnl_pct < -10:
                        signal = "alert"
                        summary_parts.append("跌幅超10%需关注")
                    elif pnl_pct > 30:
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
                info_df = ak.stock_individual_info_em(symbol=code)
                info = dict(zip(info_df["item"], info_df["value"]))
                price = float(info.get("最新", 0) or 0)

                # 计算 TTM 股息率
                ttm_yield = None
                try:
                    div_df = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
                    if not div_df.empty and price > 0:
                        div_df["除权除息日"] = pd.to_datetime(div_df["除权除息日"], errors="coerce")
                        cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
                        recent = div_df[div_df["除权除息日"] >= cutoff]
                        if not recent.empty:
                            total = recent["派息(每10股税前)"].astype(float).sum()
                            ttm_yield = round(total / 10 / price * 100, 2)
                except Exception:
                    pass

                signal = "normal"
                summary_parts = [f"现价{price}"]

                if ttm_yield is not None:
                    summary_parts.append(f"TTM股息率{ttm_yield}%")
                    if w.get("alert_yield") and ttm_yield >= w["alert_yield"]:
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
        known_codes = set()
        for p in self.memory.decisions.get_positions():
            known_codes.add(p["stock_code"])
        for w in self.memory.decisions.get_watchlist():
            known_codes.add(w["stock_code"])

        # ── 维度1：AH折价（溢价率<80的，即A股比H股便宜）──
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

    # ── LLM 生成综合分析 ─────────────────────

    def _generate_summary(self, pos_results: list, watch_results: list,
                          disc_results: list) -> str:
        all_data = json.dumps(
            {"持仓": pos_results, "自选": watch_results, "发现": disc_results},
            ensure_ascii=False, indent=2,
        )
        try:
            resp = self.llm.chat(
                messages=[Message(role="user", text=(
                    f"以下是今日A股扫描数据，请生成150字以内的今日要点，"
                    f"给出明确判断，不罗列数据：\n\n{all_data}"
                ))],
                tools=[],
                system="你是一个投资分析师，输出简洁的每日要点总结。",
            )
            return resp.text or "摘要生成失败"
        except Exception as e:
            return f"摘要生成失败: {e}"

    # ── 主入口 ────────────────────────────────

    def run(self):
        print(f"[{datetime.now().isoformat()}] 开始每日扫描...")

        pos_results = self.scan_positions()
        watch_results = self.scan_watchlist()
        disc_results = self.scan_discovery()

        summary = self._generate_summary(pos_results, watch_results, disc_results)
        print(f"  综合分析: {summary[:100]}...")

        # 发邮件
        try:
            send_report(
                summary=summary,
                positions=pos_results,
                watchlist=watch_results,
                discovery=disc_results,
            )
            print("  邮件发送成功")
        except Exception as e:
            print(f"  邮件发送失败: {e}")

        # 结果写入 scan_results 表
        for item in pos_results + watch_results + disc_results:
            self.memory.decisions.save_scan_result(item)

        print(f"[{datetime.now().isoformat()}] 扫描完成")


def main():
    parser = argparse.ArgumentParser(description="A股投资助理 - 定时扫描")
    parser.add_argument("--now", action="store_true", help="立即执行一次扫描")
    args = parser.parse_args()

    scanner = DailyScanner()

    if args.now:
        scanner.run()
        return

    print("定时任务已启动，每天 15:30 执行扫描...")
    schedule.every().day.at("15:30").do(scanner.run)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
