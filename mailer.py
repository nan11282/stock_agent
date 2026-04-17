"""
mailer.py -- 邮件发送 + HTML 模板

SMTP 配置全部从环境变量读取：
  MAIL_HOST   smtp.qq.com
  MAIL_PORT   465
  MAIL_USER   发件邮箱
  MAIL_PASS   授权码（非登录密码）
  MAIL_TO     收件邮箱
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date


MAIL_HOST = os.environ.get("MAIL_HOST", "smtp.qq.com")
MAIL_PORT = int(os.environ.get("MAIL_PORT", "465"))
MAIL_USER = os.environ.get("MAIL_USER", "")
MAIL_PASS = os.environ.get("MAIL_PASS", "")
MAIL_TO = os.environ.get("MAIL_TO", "")


def _build_html(summary: str, positions: list[dict],
                watchlist: list[dict], discovery: list[dict]) -> str:
    today = date.today().isoformat()

    # 信号标签
    def signal_tag(signal: str, scope: str = "") -> str:
        if signal == "alert":
            return '<span style="color:#d32f2f;font-weight:bold">⚠ 需关注</span>'
        if signal == "opportunity":
            return '<span style="color:#1976d2;font-weight:bold">★ 机会</span>'
        return '<span style="color:#388e3c">✓ 正常</span>'

    # ── 持仓表格 ──
    pos_rows = ""
    for p in positions:
        pos_rows += f"""<tr>
            <td>{p.get('stock_name','')}</td>
            <td>{p.get('stock_code','')}</td>
            <td>{p.get('summary','')}</td>
            <td>{signal_tag(p.get('signal','normal'))}</td>
        </tr>"""

    # ── 自选表格 ──
    watch_rows = ""
    for w in watchlist:
        tag = "★ 触发" if w.get("signal") == "alert" else "✓ 等待"
        color = "#d32f2f" if w.get("signal") == "alert" else "#388e3c"
        watch_rows += f"""<tr>
            <td>{w.get('stock_name','')}</td>
            <td>{w.get('stock_code','')}</td>
            <td>{w.get('summary','')}</td>
            <td><span style="color:{color};font-weight:bold">{tag}</span></td>
        </tr>"""

    # ── 发现表格 ──
    disc_rows = ""
    for d in discovery:
        disc_rows += f"""<tr>
            <td>{d.get('stock_name','')}</td>
            <td>{d.get('stock_code','')}</td>
            <td>{d.get('summary','')}</td>
            <td>{signal_tag(d.get('signal','normal'))}</td>
        </tr>"""

    table_style = (
        'style="border-collapse:collapse;width:100%;margin:12px 0" '
        'cellpadding="8" cellspacing="0"'
    )
    th_style = 'style="background:#f5f5f5;border:1px solid #ddd;text-align:left"'
    td_style = 'style="border:1px solid #ddd"'

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto;padding:20px">

<h2 style="color:#1a237e;border-bottom:2px solid #1a237e;padding-bottom:8px">
    A股投资助理 · 每日报告
</h2>
<p style="color:#666">{today}</p>

<h3 style="color:#333">📊 综合分析</h3>
<div style="background:#e8f5e9;padding:12px;border-radius:6px;margin:8px 0">
    {summary}
</div>

<h3 style="color:#333">💼 持仓扫描</h3>
<table {table_style}>
    <tr>
        <th {th_style}>股票</th>
        <th {th_style}>代码</th>
        <th {th_style}>摘要</th>
        <th {th_style}>状态</th>
    </tr>
    {pos_rows or '<tr><td colspan="4" style="border:1px solid #ddd;color:#999;text-align:center">暂无持仓</td></tr>'}
</table>

<h3 style="color:#333">👀 自选监控</h3>
<table {table_style}>
    <tr>
        <th {th_style}>股票</th>
        <th {th_style}>代码</th>
        <th {th_style}>摘要</th>
        <th {th_style}>状态</th>
    </tr>
    {watch_rows or '<tr><td colspan="4" style="border:1px solid #ddd;color:#999;text-align:center">暂无自选</td></tr>'}
</table>

<h3 style="color:#333">🔍 市场发现</h3>
<table {table_style}>
    <tr>
        <th {th_style}>来源</th>
        <th {th_style}>代码</th>
        <th {th_style}>摘要</th>
        <th {th_style}>状态</th>
    </tr>
    {disc_rows or '<tr><td colspan="4" style="border:1px solid #ddd;color:#999;text-align:center">暂无发现</td></tr>'}
</table>

<hr style="margin-top:24px;border:none;border-top:1px solid #ddd">
<p style="color:#999;font-size:12px">
    免责声明：本报告由AI生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。
</p>

</body>
</html>"""
    return html


def send_report(summary: str, positions: list[dict],
                watchlist: list[dict], discovery: list[dict]) -> None:
    if not MAIL_USER or not MAIL_PASS or not MAIL_TO:
        print("  [邮件] 未配置 MAIL_USER / MAIL_PASS / MAIL_TO，跳过发送")
        return

    today = date.today().isoformat()
    alert_count = sum(1 for r in positions + watchlist if r.get("signal") == "alert")
    opp_count = sum(1 for r in discovery if r.get("signal") == "opportunity")

    subject = f"【A股助理】{today} 每日报告"
    if alert_count:
        subject += f" ⚠ {alert_count}条提醒"
    if opp_count:
        subject += f" ★ {opp_count}个机会"

    html = _build_html(summary, positions, watchlist, discovery)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_TO
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP_SSL(MAIL_HOST, MAIL_PORT) as server:
        server.login(MAIL_USER, MAIL_PASS)
        server.sendmail(MAIL_USER, MAIL_TO.split(","), msg.as_string())
