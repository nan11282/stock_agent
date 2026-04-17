"""
memory.py -- 记忆系统

SQLite   : 结构化决策日志 / 自选股 / 持仓 / 复盘 / 对话摘要 + FTS5 全文检索
ChromaDB : 向量语义检索（HNSW 近似最近邻）
RRF      : Reciprocal Rank Fusion 融合向量 + FTS5 两路结果
"""

# ── pysqlite3 shim（Docker 里使用，本地无影响）──
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import sqlite3
import json
import uuid
from datetime import datetime


# ── 路径配置 ──────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "./data/investment.db")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")


# ─────────────────────────────────────────────
# SQLite -- 决策日志 / 自选股 / 持仓 / 复盘
# ─────────────────────────────────────────────

class DecisionLog:
    def __init__(self, db_path: str = None):
        db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            -- 投资决策记录（append-only，不允许 UPDATE）
            CREATE TABLE IF NOT EXISTS decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                stock_code  TEXT NOT NULL,
                stock_name  TEXT,
                action      TEXT,
                view        TEXT,
                reasoning   TEXT NOT NULL,
                price       REAL,
                ttm_yield   REAL,
                pe_pct      REAL,
                pe_abs      REAL,
                tags        TEXT
            );

            -- 自选股关注列表
            CREATE TABLE IF NOT EXISTS watchlist (
                stock_code   TEXT PRIMARY KEY,
                stock_name   TEXT NOT NULL,
                reason       TEXT,
                alert_yield  REAL,
                alert_pe_pct REAL,
                added_at     TEXT NOT NULL
            );

            -- 持仓表
            CREATE TABLE IF NOT EXISTS positions (
                stock_code   TEXT PRIMARY KEY,
                stock_name   TEXT NOT NULL,
                cost_price   REAL NOT NULL,
                shares       INTEGER,
                position_pct REAL,
                tier         TEXT,
                updated_at   TEXT NOT NULL
            );

            -- 复盘表（挂在 decisions 下，不修改原始记录）
            CREATE TABLE IF NOT EXISTS retrospectives (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id     INTEGER NOT NULL REFERENCES decisions(id),
                reviewed_at     TEXT NOT NULL,
                price_now       REAL,
                outcome         TEXT,
                what_i_missed   TEXT,
                updated_view    TEXT
            );

            -- 对话摘要存储（供向量检索使用的原始文本）
            CREATE TABLE IF NOT EXISTS episodic_docs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      TEXT UNIQUE NOT NULL,
                text        TEXT NOT NULL,
                metadata    TEXT,
                created_at  TEXT NOT NULL
            );

            -- 每日扫描结果
            CREATE TABLE IF NOT EXISTS scan_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at  TEXT NOT NULL,
                scope       TEXT,
                stock_code  TEXT,
                stock_name  TEXT,
                signal      TEXT,
                summary     TEXT
            );
        """)
        # FTS5 虚拟表单独建（不能在 executescript 的事务里和其他语句混用）
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
                USING fts5(text, doc_id UNINDEXED, content='episodic_docs', content_rowid='id')
        """)
        self.conn.commit()

    # ── 读操作（Agent 可自主调用）────────────────

    def search_decisions(self, stock_code: str = None,
                         keyword: str = None, limit: int = 10) -> list[dict]:
        query = "SELECT * FROM decisions WHERE 1=1"
        params: list = []
        if stock_code:
            query += " AND stock_code = ?"
            params.append(stock_code)
        if keyword:
            query += " AND (reasoning LIKE ? OR stock_name LIKE ? OR tags LIKE ?)"
            params += [f"%{keyword}%"] * 3
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_decision_by_id(self, decision_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM decisions WHERE id=?", (decision_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_positions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM positions ORDER BY position_pct DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_watchlist(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM watchlist ORDER BY added_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def search_retrospectives(self, decision_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM retrospectives WHERE decision_id=? ORDER BY reviewed_at DESC",
            (decision_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 写操作（必须经用户确认后才能调用）────────

    def save_decision(self, data: dict) -> int:
        cur = self.conn.execute("""
            INSERT INTO decisions
            (created_at, stock_code, stock_name, action, view,
             reasoning, price, ttm_yield, pe_pct, pe_abs, tags)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            data["stock_code"],
            data.get("stock_name"),
            data.get("action"),
            data.get("view"),
            data["reasoning"],
            data.get("price"),
            data.get("ttm_yield"),
            data.get("pe_pct"),
            data.get("pe_abs"),
            json.dumps(data.get("tags", []), ensure_ascii=False),
        ))
        self.conn.commit()
        return cur.lastrowid

    def delete_decision(self, decision_id: int) -> bool:
        affected = self.conn.execute(
            "DELETE FROM decisions WHERE id=?", (decision_id,)
        ).rowcount
        self.conn.commit()
        return affected > 0

    def save_retrospective(self, data: dict) -> int:
        cur = self.conn.execute("""
            INSERT INTO retrospectives
            (decision_id, reviewed_at, price_now, outcome, what_i_missed, updated_view)
            VALUES (?,?,?,?,?,?)
        """, (
            data["decision_id"],
            datetime.now().isoformat(),
            data.get("price_now"),
            data.get("outcome"),
            data.get("what_i_missed"),
            data.get("updated_view"),
        ))
        self.conn.commit()
        return cur.lastrowid

    def upsert_position(self, data: dict) -> None:
        self.conn.execute("""
            INSERT INTO positions
            (stock_code, stock_name, cost_price, shares, position_pct, tier, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(stock_code) DO UPDATE SET
                stock_name   = excluded.stock_name,
                cost_price   = excluded.cost_price,
                shares       = excluded.shares,
                position_pct = excluded.position_pct,
                tier         = excluded.tier,
                updated_at   = excluded.updated_at
        """, (
            data["stock_code"],
            data["stock_name"],
            data["cost_price"],
            data.get("shares"),
            data.get("position_pct"),
            data.get("tier"),
            datetime.now().isoformat(),
        ))
        self.conn.commit()

    def delete_position(self, stock_code: str) -> bool:
        affected = self.conn.execute(
            "DELETE FROM positions WHERE stock_code=?", (stock_code,)
        ).rowcount
        self.conn.commit()
        return affected > 0

    def upsert_watchlist(self, data: dict) -> None:
        self.conn.execute("""
            INSERT INTO watchlist
            (stock_code, stock_name, reason, alert_yield, alert_pe_pct, added_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(stock_code) DO UPDATE SET
                stock_name   = excluded.stock_name,
                reason       = excluded.reason,
                alert_yield  = excluded.alert_yield,
                alert_pe_pct = excluded.alert_pe_pct,
                added_at     = excluded.added_at
        """, (
            data["stock_code"],
            data["stock_name"],
            data.get("reason"),
            data.get("alert_yield"),
            data.get("alert_pe_pct"),
            datetime.now().isoformat(),
        ))
        self.conn.commit()

    def delete_watchlist(self, stock_code: str) -> bool:
        affected = self.conn.execute(
            "DELETE FROM watchlist WHERE stock_code=?", (stock_code,)
        ).rowcount
        self.conn.commit()
        return affected > 0

    # ── 扫描结果写入 ──────────────────────────

    def save_scan_result(self, data: dict) -> int:
        cur = self.conn.execute("""
            INSERT INTO scan_results
            (scanned_at, scope, stock_code, stock_name, signal, summary)
            VALUES (?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            data.get("scope"),
            data.get("stock_code"),
            data.get("stock_name"),
            data.get("signal"),
            data.get("summary"),
        ))
        self.conn.commit()
        return cur.lastrowid


# ─────────────────────────────────────────────
# EpisodicMemory -- 向量 + FTS5 混合检索
# ─────────────────────────────────────────────

class EpisodicMemory:
    """
    存储结构:
      ChromaDB  : text + embedding 向量 → 语义检索（HNSW 近似最近邻）
      SQLite FTS5 : text 的倒排索引 → 精确词检索（对股票代码/数字敏感）
      两者通过 doc_id 关联，写入时同步，检索时独立查询后 RRF 融合
    """

    def __init__(self, db_path: str = None, persist_dir: str = None):
        import chromadb

        db_path = db_path or DB_PATH
        persist_dir = persist_dir or CHROMA_PATH

        # SQLite 连接（复用 DecisionLog 同一个数据库文件）
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="investment_memory",
            metadata={"hnsw:space": "cosine"},
        )

    # ── 分词（中文 jieba）──────────────────────

    @staticmethod
    def _tokenize(text: str) -> str:
        """用 jieba 分词，返回空格连接的 token 字符串（给 FTS5 MATCH 用）"""
        import jieba
        tokens = jieba.cut(text)
        return " ".join(t.strip() for t in tokens if t.strip())

    # ── 写入 ────────────────────────────────────

    def save_insight(self, text: str, metadata: dict = None) -> str:
        doc_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        meta = {**(metadata or {}), "saved_at": now}

        # 1. 写入 ChromaDB（自动生成 embedding 向量）
        self.collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[doc_id],
        )

        # 2. 写入 SQLite episodic_docs 表
        self.conn.execute("""
            INSERT INTO episodic_docs (doc_id, text, metadata, created_at)
            VALUES (?, ?, ?, ?)
        """, (doc_id, text, json.dumps(meta, ensure_ascii=False), now))

        # 3. 更新 FTS5 索引（存分词后的文本，便于中文检索）
        rowid = self.conn.execute(
            "SELECT id FROM episodic_docs WHERE doc_id=?", (doc_id,)
        ).fetchone()[0]
        self.conn.execute(
            "INSERT INTO episodic_fts(rowid, text, doc_id) VALUES (?, ?, ?)",
            (rowid, self._tokenize(text), doc_id),
        )
        self.conn.commit()
        return doc_id

    # ── 混合检索主入口 ──────────────────────────

    def retrieve(self, query: str, n_results: int = 8, top_k: int = 4) -> list[dict]:
        if self.collection.count() == 0:
            return []

        n = min(n_results, self.collection.count())

        # ── 路1：ChromaDB 向量检索（HNSW 近似最近邻，语义感知）──
        vec_result = self.collection.query(query_texts=[query], n_results=n)
        vec_ids: list[str] = vec_result["ids"][0]

        # ── 路2：SQLite FTS5 全文检索（精确词匹配，对股票代码/数字敏感）──
        fts_ids: list[str] = []
        try:
            tokenized_query = self._tokenize(query)
            # 用 OR 连接各 token，提高中文召回率
            match_expr = " OR ".join(tokenized_query.split())
            if match_expr.strip():
                rows = self.conn.execute(
                    "SELECT doc_id, rank FROM episodic_fts WHERE text MATCH ? ORDER BY rank LIMIT ?",
                    (match_expr, n),
                ).fetchall()
                fts_ids = [row["doc_id"] for row in rows]
        except Exception:
            pass  # FTS 查询失败不影响向量检索结果

        # ── 路3：RRF 融合（K=60）──
        K = 60
        rrf_scores: dict[str, float] = {}

        for rank, doc_id in enumerate(vec_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (K + rank + 1)

        for rank, doc_id in enumerate(fts_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (K + rank + 1)

        # ── 取 top-k ──
        top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        if not top_ids:
            return []

        fetched = self.collection.get(ids=top_ids, include=["documents", "metadatas"])

        return [
            {
                "text": doc,
                "metadata": meta,
                "rrf_score": round(rrf_scores[id_], 6),
            }
            for id_, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        ]


# ─────────────────────────────────────────────
# MemoryManager -- 统一入口
# ─────────────────────────────────────────────

class MemoryManager:
    def __init__(self):
        self.decisions = DecisionLog()
        self.episodic = EpisodicMemory()

    def retrieve_context(self, user_query: str) -> str:
        fragments = self.episodic.retrieve(user_query, n_results=8, top_k=4)
        decision_hits = self.decisions.search_decisions(keyword=user_query, limit=3)

        parts = []

        if fragments:
            parts.append("【相关历史洞察（按相关度排列）】")
            for f in fragments:
                score_str = f"score={f['rrf_score']:.4f}"
                parts.append(f"- [{score_str}] {f['text'][:200]}")

        if decision_hits:
            parts.append("\n【相关历史决策记录】")
            for d in decision_hits:
                parts.append(
                    f"- [id={d['id']} | {d['created_at'][:10]}] "
                    f"{d['stock_name']}({d['stock_code']}) "
                    f"观点:{d['view']} | {d['reasoning'][:120]}"
                )

        return "\n".join(parts) if parts else ""
