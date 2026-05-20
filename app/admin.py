"""
admin.py -- 本地后台管理

目标是给“记忆/持仓/自选/决策日志”提供一个低依赖、可审计的管理入口。
不引入前端构建链，也不把管理端写入绕过 memory.py 的业务语义。
"""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from urllib.parse import parse_qs, quote, urlparse

from memory import CHROMA_PATH, DB_PATH, DecisionLog, EpisodicMemory


ADMIN_HOST = os.environ.get("ADMIN_HOST", "127.0.0.1")
ADMIN_PORT = int(os.environ.get("ADMIN_PORT", "8787"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
ADMIN_ALLOW_NO_AUTH = os.environ.get("ADMIN_ALLOW_NO_AUTH", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


def _form_value(form: dict[str, list[str]], name: str, default: str = "") -> str:
    return (form.get(name) or [default])[0].strip()


def _json_metadata(raw: str) -> dict:
    if not raw.strip():
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("metadata 必须是 JSON object")
    return decoded


class AdminStore:
    def __init__(self, db_path: str = DB_PATH, chroma_path: str = CHROMA_PATH):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.decisions = DecisionLog(db_path=db_path)
        self._episodic: EpisodicMemory | None = None

    @property
    def episodic(self) -> EpisodicMemory:
        if self._episodic is None:
            self._episodic = EpisodicMemory(
                db_path=self.db_path,
                persist_dir=self.chroma_path,
            )
        return self._episodic

    def list_insights(self, keyword: str = "", limit: int = 80) -> list[dict]:
        conn = self.decisions.conn
        params: list = []
        query = "SELECT * FROM episodic_docs"
        if keyword:
            query += " WHERE text LIKE ? OR metadata LIKE ? OR doc_id LIKE ?"
            params.extend([f"%{keyword}%"] * 3)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [EpisodicMemory._format_insight_row(row) for row in rows]

    def get_insight(self, doc_id: str) -> dict | None:
        row = self.decisions.conn.execute(
            "SELECT * FROM episodic_docs WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return EpisodicMemory._format_insight_row(row) if row else None

    def save_insight(
        self, *, text: str, metadata: dict | None = None, doc_id: str | None = None
    ) -> str:
        if not text.strip():
            raise ValueError("记忆文本不能为空")
        if doc_id:
            updated = self.episodic.update_insight(doc_id, text, metadata or {})
            if not updated:
                raise ValueError(f"找不到记忆: {doc_id}")
            return doc_id
        return self.episodic.save_insight(text, metadata or {})

    def delete_insight(self, doc_id: str) -> bool:
        return self.episodic.delete_insight(doc_id)

    def list_decisions(self, keyword: str = "", limit: int = 80) -> list[dict]:
        return self.decisions.search_decisions(keyword=keyword or None, limit=limit)

    def list_retrospectives(self, decision_id: int) -> list[dict]:
        return self.decisions.search_retrospectives(decision_id)

    def list_scans(self, limit: int = 30) -> list[dict]:
        rows = self.decisions.conn.execute(
            "SELECT * FROM scan_results ORDER BY scanned_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


class AdminHandler(BaseHTTPRequestHandler):
    store: AdminStore

    def log_message(self, fmt: str, *args) -> None:
        print(f"[admin] {self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self) -> None:
        try:
            self._dispatch_get()
        except Exception as e:
            self._render("错误", self._notice(str(e), "error"), status=500)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/login":
                form = self._read_form()
                self._post_login(form)
                return
            if not self._is_authenticated():
                self._redirect("/login")
                return
            form = self._read_form()
            if path == "/memory/save":
                self._post_memory_save(form)
            elif path == "/memory/delete":
                self._post_memory_delete(form)
            elif path == "/position/save":
                self._post_position_save(form)
            elif path == "/position/delete":
                self._post_position_delete(form)
            elif path == "/watchlist/save":
                self._post_watchlist_save(form)
            elif path == "/watchlist/delete":
                self._post_watchlist_delete(form)
            elif path == "/decision/save":
                self._post_decision_save(form)
            elif path == "/decision/delete":
                self._post_decision_delete(form)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as e:
            self._render("错误", self._notice(str(e), "error"), status=400)

    def _dispatch_get(self) -> None:
        path = urlparse(self.path).path
        if path == "/login":
            self._login_page()
            return
        if path == "/logout":
            self._logout()
            return
        if not self._is_authenticated():
            self._redirect("/login")
            return
        if path == "/":
            self._dashboard()
        elif path == "/memories":
            self._memories_page()
        elif path == "/memory":
            self._memory_edit_page()
        elif path == "/positions":
            self._positions_page()
        elif path == "/watchlist":
            self._watchlist_page()
        elif path == "/decisions":
            self._decisions_page()
        elif path == "/scans":
            self._scans_page()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _read_form(self) -> dict[str, list[str]]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return parse_qs(raw, keep_blank_values=True)

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query, keep_blank_values=True)

    def _is_authenticated(self) -> bool:
        if not ADMIN_TOKEN:
            return True
        query_token = (self._query().get("token") or [""])[0]
        if query_token and query_token == ADMIN_TOKEN:
            return True
        cookie = SimpleCookie(self.headers.get("Cookie"))
        morsel = cookie.get("admin_token")
        return bool(morsel and morsel.value == ADMIN_TOKEN)

    def _post_login(self, form: dict[str, list[str]]) -> None:
        token = _form_value(form, "token")
        if ADMIN_TOKEN and token != ADMIN_TOKEN:
            self._render("登录", self._notice("口令不正确", "error") + self._login_form())
            return
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/")
        if ADMIN_TOKEN:
            self.send_header(
                "Set-Cookie",
                f"admin_token={quote(token)}; HttpOnly; SameSite=Lax; Path=/",
            )
        self.end_headers()

    def _logout(self) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/login")
        self.send_header(
            "Set-Cookie",
            "admin_token=; Max-Age=0; HttpOnly; SameSite=Lax; Path=/",
        )
        self.end_headers()

    def _post_memory_save(self, form: dict[str, list[str]]) -> None:
        doc_id = _form_value(form, "doc_id") or None
        text = _form_value(form, "text")
        metadata = _json_metadata(_form_value(form, "metadata"))
        saved_doc_id = self.store.save_insight(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
        )
        self._redirect(f"/memories?notice={quote('记忆已保存: ' + saved_doc_id)}")

    def _post_memory_delete(self, form: dict[str, list[str]]) -> None:
        doc_id = _form_value(form, "doc_id")
        ok = self.store.delete_insight(doc_id)
        notice = "记忆已删除" if ok else "记忆不存在"
        self._redirect(f"/memories?notice={quote(notice)}")

    def _post_position_save(self, form: dict[str, list[str]]) -> None:
        data = {
            "stock_code": _form_value(form, "stock_code"),
            "stock_name": _form_value(form, "stock_name"),
            "cost_price": _to_float(_form_value(form, "cost_price")),
            "shares": _to_int(_form_value(form, "shares")),
            "position_pct": _to_float(_form_value(form, "position_pct")),
            "tier": _form_value(form, "tier"),
        }
        if not data["stock_code"] or not data["stock_name"] or data["cost_price"] is None:
            raise ValueError("持仓需要 stock_code、stock_name、cost_price")
        self.store.decisions.upsert_position(data)
        self._redirect("/positions?notice=%E6%8C%81%E4%BB%93%E5%B7%B2%E4%BF%9D%E5%AD%98")

    def _post_position_delete(self, form: dict[str, list[str]]) -> None:
        self.store.decisions.delete_position(_form_value(form, "stock_code"))
        self._redirect("/positions?notice=%E6%8C%81%E4%BB%93%E5%B7%B2%E5%88%A0%E9%99%A4")

    def _post_watchlist_save(self, form: dict[str, list[str]]) -> None:
        data = {
            "stock_code": _form_value(form, "stock_code"),
            "stock_name": _form_value(form, "stock_name"),
            "reason": _form_value(form, "reason"),
            "alert_yield": _to_float(_form_value(form, "alert_yield")),
            "alert_pe_pct": _to_float(_form_value(form, "alert_pe_pct")),
            "alert_price_below": _to_float(_form_value(form, "alert_price_below")),
            "watch_price_below": _to_float(_form_value(form, "watch_price_below")),
            "alert_note": _form_value(form, "alert_note"),
        }
        if not data["stock_code"] or not data["stock_name"]:
            raise ValueError("自选股需要 stock_code 和 stock_name")
        self.store.decisions.upsert_watchlist(data)
        self._redirect("/watchlist?notice=%E8%87%AA%E9%80%89%E5%B7%B2%E4%BF%9D%E5%AD%98")

    def _post_watchlist_delete(self, form: dict[str, list[str]]) -> None:
        self.store.decisions.delete_watchlist(_form_value(form, "stock_code"))
        self._redirect("/watchlist?notice=%E8%87%AA%E9%80%89%E5%B7%B2%E5%88%A0%E9%99%A4")

    def _post_decision_save(self, form: dict[str, list[str]]) -> None:
        data = {
            "stock_code": _form_value(form, "stock_code"),
            "stock_name": _form_value(form, "stock_name"),
            "action": _form_value(form, "action"),
            "view": _form_value(form, "view"),
            "reasoning": _form_value(form, "reasoning"),
            "price": _to_float(_form_value(form, "price")),
            "ttm_yield": _to_float(_form_value(form, "ttm_yield")),
            "pe_pct": _to_float(_form_value(form, "pe_pct")),
            "pe_abs": _to_float(_form_value(form, "pe_abs")),
            "tags": [
                tag.strip()
                for tag in _form_value(form, "tags").split(",")
                if tag.strip()
            ],
        }
        if not data["stock_code"] or not data["reasoning"]:
            raise ValueError("决策日志需要 stock_code 和 reasoning")
        new_id = self.store.decisions.save_decision(data)
        self._redirect(f"/decisions?notice={quote('决策已保存: ' + str(new_id))}")

    def _post_decision_delete(self, form: dict[str, list[str]]) -> None:
        self.store.decisions.delete_decision(int(_form_value(form, "decision_id")))
        self._redirect("/decisions?notice=%E5%86%B3%E7%AD%96%E5%B7%B2%E5%88%A0%E9%99%A4")

    def _dashboard(self) -> None:
        conn = self.store.decisions.conn
        counts = {}
        for table in (
            "episodic_docs",
            "decisions",
            "positions",
            "watchlist",
            "retrospectives",
            "scan_results",
        ):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        cards = "".join(
            f"""
            <a class="metric" href="{href}">
              <span>{label}</span>
              <strong>{counts[key]}</strong>
            </a>
            """
            for key, label, href in (
                ("episodic_docs", "长期洞察", "/memories"),
                ("decisions", "决策日志", "/decisions"),
                ("positions", "持仓", "/positions"),
                ("watchlist", "自选", "/watchlist"),
                ("retrospectives", "复盘", "/decisions"),
                ("scan_results", "扫描快照", "/scans"),
            )
        )
        body = f"""
        <section class="grid metrics">{cards}</section>
        <section>
          <h2>管理边界</h2>
          <p class="muted">这里直接管理长期记忆和组合事实。投资判断仍建议保留追加式日志，避免事后覆盖历史推理。</p>
        </section>
        """
        self._render("后台概览", body)

    def _memories_page(self) -> None:
        query = _form_value(self._query(), "q")
        notice = _form_value(self._query(), "notice")
        rows = self.store.list_insights(query)
        items = "".join(self._memory_item(row) for row in rows)
        body = self._notice(notice) if notice else ""
        body += f"""
        <section class="toolbar">
          <form method="get" action="/memories">
            <input name="q" value="{escape(query)}" placeholder="搜索文本、metadata、doc_id">
            <button>搜索</button>
            <a class="button secondary" href="/memories">清空</a>
          </form>
          <a class="button secondary" href="/memory">完整编辑页</a>
        </section>
        <form class="panel" method="post" action="/memory/save">
          <h2>新增记忆</h2>
          <textarea name="text" rows="5" placeholder="输入要沉淀的长期洞察，例如：高股息银行仓位只在低估且分红稳定时加仓。" required></textarea>
          <label>metadata JSON</label>
          <textarea name="metadata" rows="4">{"{}"}</textarea>
          <div class="actions">
            <button>新增到长期记忆</button>
          </div>
        </form>
        <section class="list">{items or '<p class="empty">暂无长期洞察。</p>'}</section>
        """
        self._render("长期洞察", body)

    def _memory_edit_page(self) -> None:
        doc_id = _form_value(self._query(), "doc_id")
        row = self.store.get_insight(doc_id) if doc_id else None
        if doc_id and row is None:
            self._render("编辑记忆", self._notice(f"找不到记忆: {doc_id}", "error"))
            return
        metadata = json.dumps(row["metadata"], ensure_ascii=False, indent=2) if row else "{}"
        title = "修改记忆" if row else "新增记忆"
        body = f"""
        <form class="panel" method="post" action="/memory/save">
          <h2>{title}</h2>
          <input type="hidden" name="doc_id" value="{escape(doc_id)}">
          <label>doc_id</label>
          <input value="{escape(doc_id or '新增后自动生成')}" disabled>
          <label>文本</label>
          <textarea name="text" rows="10" required>{escape(row["text"] if row else "")}</textarea>
          <label>metadata JSON</label>
          <textarea name="metadata" rows="7">{escape(metadata)}</textarea>
          <div class="actions">
            <button>保存</button>
            <a class="button secondary" href="/memories">返回</a>
          </div>
        </form>
        """
        self._render("编辑记忆", body)

    def _memory_item(self, row: dict) -> str:
        meta = json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True)
        return f"""
        <article class="item">
          <div class="item-head">
            <span class="mono">{escape(row["doc_id"])}</span>
            <span>{escape(row["created_at"])}</span>
          </div>
          <p class="memory-text">{escape(row["text"])}</p>
          <pre>{escape(meta)}</pre>
          <div class="item-actions">
            <a class="button secondary" href="/memory?doc_id={quote(row["doc_id"])}">编辑</a>
            <form method="post" action="/memory/delete" onsubmit="return confirm('确认删除这条长期记忆？')">
              <input type="hidden" name="doc_id" value="{escape(row["doc_id"])}">
              <button class="danger">删除</button>
            </form>
          </div>
        </article>
        """

    def _positions_page(self) -> None:
        rows = self.store.decisions.get_positions()
        body = self._notice(_form_value(self._query(), "notice"))
        body += self._position_form()
        body += "<section class='table-wrap'>" + self._table(
            rows,
            ["stock_code", "stock_name", "cost_price", "shares", "position_pct", "tier", "updated_at"],
            delete_action="/position/delete",
            delete_key="stock_code",
        ) + "</section>"
        self._render("持仓管理", body)

    def _position_form(self) -> str:
        return """
        <form class="panel compact" method="post" action="/position/save">
          <div class="form-grid">
            <input name="stock_code" placeholder="代码" required>
            <input name="stock_name" placeholder="名称" required>
            <input name="cost_price" placeholder="成本价" required>
            <input name="shares" placeholder="股数">
            <input name="position_pct" placeholder="仓位%">
            <select name="tier"><option value="">分层</option><option value="core">core</option><option value="growth">growth</option></select>
          </div>
          <button>保存持仓</button>
        </form>
        """

    def _watchlist_page(self) -> None:
        rows = self.store.decisions.get_watchlist()
        body = self._notice(_form_value(self._query(), "notice"))
        body += """
        <form class="panel compact" method="post" action="/watchlist/save">
          <div class="form-grid">
            <input name="stock_code" placeholder="代码" required>
            <input name="stock_name" placeholder="名称" required>
            <input name="reason" placeholder="关注原因">
            <input name="alert_yield" placeholder="股息率阈值%">
            <input name="alert_pe_pct" placeholder="PE百分位阈值">
            <input name="alert_price_below" placeholder="强提醒价">
            <input name="watch_price_below" placeholder="观察价">
            <input name="alert_note" placeholder="提醒备注">
          </div>
          <button>保存自选</button>
        </form>
        """
        body += "<section class='table-wrap'>" + self._table(
            rows,
            [
                "stock_code",
                "stock_name",
                "reason",
                "alert_yield",
                "alert_pe_pct",
                "alert_price_below",
                "watch_price_below",
                "alert_note",
                "added_at",
            ],
            delete_action="/watchlist/delete",
            delete_key="stock_code",
        ) + "</section>"
        self._render("自选管理", body)

    def _decisions_page(self) -> None:
        query = _form_value(self._query(), "q")
        rows = self.store.list_decisions(keyword=query)
        body = self._notice(_form_value(self._query(), "notice"))
        body += f"""
        <section class="toolbar">
          <form method="get" action="/decisions">
            <input name="q" value="{escape(query)}" placeholder="搜索 reasoning、名称、tags">
            <button>搜索</button>
          </form>
        </section>
        <form class="panel" method="post" action="/decision/save">
          <div class="form-grid">
            <input name="stock_code" placeholder="代码" required>
            <input name="stock_name" placeholder="名称">
            <select name="action"><option value="">action</option><option>analysis</option><option>watch</option><option>hold</option><option>buy_signal</option><option>sell_signal</option></select>
            <select name="view"><option value="">view</option><option>bullish</option><option>neutral</option><option>bearish</option></select>
            <input name="price" placeholder="价格">
            <input name="ttm_yield" placeholder="股息率">
            <input name="pe_pct" placeholder="PE百分位">
            <input name="pe_abs" placeholder="PE绝对值">
            <input name="tags" placeholder="tags,逗号分隔">
          </div>
          <textarea name="reasoning" rows="5" placeholder="完整推理" required></textarea>
          <button>追加决策</button>
        </form>
        """
        body += "<section class='table-wrap'>" + self._table(
            rows,
            [
                "id",
                "created_at",
                "stock_code",
                "stock_name",
                "action",
                "view",
                "reasoning",
                "price",
                "ttm_yield",
                "pe_pct",
                "pe_abs",
                "tags",
            ],
            delete_action="/decision/delete",
            delete_key="id",
            delete_name="decision_id",
        ) + "</section>"
        self._render("决策日志", body)

    def _scans_page(self) -> None:
        rows = self.store.list_scans()
        body = "<section class='table-wrap'>" + self._table(
            rows,
            ["id", "scanned_at", "scope", "stock_code", "stock_name", "signal", "summary"],
        ) + "</section>"
        self._render("扫描快照", body)

    def _table(
        self,
        rows: list[dict],
        columns: list[str],
        delete_action: str | None = None,
        delete_key: str | None = None,
        delete_name: str | None = None,
    ) -> str:
        if not rows:
            return "<p class='empty'>暂无数据。</p>"
        header = "".join(f"<th>{escape(col)}</th>" for col in columns)
        if delete_action:
            header += "<th>操作</th>"
        body_rows = []
        for row in rows:
            cells = "".join(
                f"<td>{escape(self._cell(row.get(col)))}</td>" for col in columns
            )
            if delete_action and delete_key:
                value = row.get(delete_key)
                name = delete_name or delete_key
                cells += f"""
                <td>
                  <form method="post" action="{delete_action}" onsubmit="return confirm('确认删除？')">
                    <input type="hidden" name="{escape(name)}" value="{escape(str(value))}">
                    <button class="danger">删除</button>
                  </form>
                </td>
                """
            body_rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    @staticmethod
    def _cell(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _login_page(self) -> None:
        if self._is_authenticated():
            self._redirect("/")
            return
        self._render("登录", self._login_form(), show_nav=False)

    @staticmethod
    def _login_form() -> str:
        return """
        <form class="panel login" method="post" action="/login">
          <h2>后台登录</h2>
          <input type="password" name="token" placeholder="ADMIN_TOKEN" autofocus>
          <button>进入</button>
        </form>
        """

    @staticmethod
    def _notice(message: str, level: str = "ok") -> str:
        if not message:
            return ""
        return f"<div class='notice {escape(level)}'>{escape(message)}</div>"

    def _redirect(self, target: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", target)
        self.end_headers()

    def _render(
        self,
        title: str,
        body: str,
        status: int = 200,
        show_nav: bool = True,
    ) -> None:
        nav = """
        <nav>
          <a href="/">概览</a>
          <a href="/memories">记忆</a>
          <a href="/positions">持仓</a>
          <a href="/watchlist">自选</a>
          <a href="/decisions">决策</a>
          <a href="/scans">扫描</a>
          <a href="/logout">退出</a>
        </nav>
        """ if show_nav else ""
        html = f"""<!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>{escape(title)} - Stock Agent Admin</title>
          <style>
            :root {{
              --bg: #f6f7f8;
              --panel: #ffffff;
              --text: #20242a;
              --muted: #667085;
              --line: #d8dee6;
              --accent: #126a5a;
              --danger: #b42318;
              --shadow: 0 1px 2px rgba(16, 24, 40, .06);
            }}
            * {{ box-sizing: border-box; }}
            body {{
              margin: 0;
              background: var(--bg);
              color: var(--text);
              font: 14px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            header {{
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 16px;
              padding: 18px 28px;
              background: var(--panel);
              border-bottom: 1px solid var(--line);
              position: sticky;
              top: 0;
              z-index: 2;
            }}
            h1 {{ margin: 0; font-size: 20px; }}
            h2 {{ margin: 0 0 12px; font-size: 16px; }}
            nav {{ display: flex; flex-wrap: wrap; gap: 8px; }}
            nav a, .button, button {{
              border: 1px solid var(--line);
              border-radius: 6px;
              color: var(--text);
              background: var(--panel);
              padding: 7px 11px;
              text-decoration: none;
              cursor: pointer;
              font: inherit;
            }}
            button, .button {{ background: var(--accent); border-color: var(--accent); color: white; }}
            .secondary {{ background: var(--panel); color: var(--text); }}
            .danger {{ background: var(--danger); border-color: var(--danger); color: white; }}
            main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
            section, .panel, .item {{
              background: var(--panel);
              border: 1px solid var(--line);
              border-radius: 8px;
              box-shadow: var(--shadow);
              margin-bottom: 16px;
              padding: 16px;
            }}
            .toolbar {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
            .toolbar form {{ display: flex; gap: 8px; flex: 1; }}
            input, textarea, select {{
              width: 100%;
              border: 1px solid var(--line);
              border-radius: 6px;
              padding: 8px 10px;
              font: inherit;
              background: white;
              color: var(--text);
            }}
            textarea {{ resize: vertical; }}
            label {{ display: block; margin: 12px 0 6px; color: var(--muted); }}
            .grid {{ display: grid; gap: 12px; }}
            .metrics {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
            .metric {{ color: var(--text); text-decoration: none; }}
            .metric span {{ color: var(--muted); display: block; }}
            .metric strong {{ font-size: 28px; }}
            .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px; margin-bottom: 10px; }}
            .item-head {{ display: flex; justify-content: space-between; gap: 12px; color: var(--muted); }}
            .item-actions {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
            .memory-text {{ white-space: pre-wrap; }}
            .mono, pre {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
            pre {{ white-space: pre-wrap; background: #f1f3f5; padding: 10px; border-radius: 6px; }}
            table {{ width: 100%; border-collapse: collapse; background: white; }}
            th, td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 9px; vertical-align: top; }}
            th {{ color: var(--muted); font-weight: 600; }}
            .table-wrap {{ overflow-x: auto; padding: 0; }}
            .notice {{ border-radius: 6px; padding: 10px 12px; margin-bottom: 16px; background: #e8f3ef; border: 1px solid #b9d9cf; }}
            .notice.error {{ background: #fff0ed; border-color: #f1b4ad; }}
            .muted, .empty {{ color: var(--muted); }}
            .actions {{ display: flex; gap: 8px; margin-top: 12px; }}
            .login {{ max-width: 380px; margin: 12vh auto; }}
            @media (max-width: 720px) {{
              header {{ align-items: flex-start; flex-direction: column; padding: 14px 18px; }}
              main {{ padding: 16px; }}
              .toolbar {{ align-items: stretch; flex-direction: column; }}
              .toolbar form {{ flex-direction: column; }}
            }}
          </style>
        </head>
        <body>
          <header><h1>{escape(title)}</h1>{nav}</header>
          <main>{body}</main>
        </body>
        </html>"""
        payload = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def build_server(
    host: str = ADMIN_HOST,
    port: int = ADMIN_PORT,
    store: AdminStore | None = None,
) -> ThreadingHTTPServer:
    AdminHandler.store = store or AdminStore()
    return ThreadingHTTPServer((host, port), AdminHandler)


def main() -> None:
    if not ADMIN_TOKEN and not ADMIN_ALLOW_NO_AUTH:
        raise SystemExit(
            "ADMIN_TOKEN 未设置。若只在可信本机调试，可显式设置 ADMIN_ALLOW_NO_AUTH=true。"
        )
    server = build_server()
    print(f"后台管理已启动: http://{ADMIN_HOST}:{ADMIN_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
