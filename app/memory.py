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

from metrics import console_timer


# ── 路径配置 ──────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "./data/investment.db")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_POOL_SIZE = int(os.environ.get("RERANK_POOL_SIZE", "12"))
SQLITE_JOURNAL_MODE = os.environ.get("SQLITE_JOURNAL_MODE", "DELETE").upper()


def _enable_wal_safely(conn, label: str) -> None:
    mode = SQLITE_JOURNAL_MODE
    if mode not in {"DELETE", "WAL", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}:
        mode = "DELETE"
    try:
        # Docker Desktop 的 Windows bind mount 对 WAL/SHM 锁支持不稳定；
        # 默认用 DELETE 保可用性，需要并发优化时可显式设 SQLITE_JOURNAL_MODE=WAL。
        conn.execute(f"PRAGMA journal_mode={mode}")
    except sqlite3.OperationalError as e:
        print(
            f"  [SQLite] journal_mode={mode} 启用失败，尝试 DELETE "
            f"label={label}: {e}",
            flush=True,
        )
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.OperationalError as fallback_e:
            print(
                f"  [SQLite] DELETE journal降级失败 label={label}: {fallback_e}",
                flush=True,
            )


def _sqlite_path_diagnostics(db_path: str) -> str:
    if db_path == ":memory:":
        return "db_path=:memory:"

    abs_path = os.path.abspath(db_path)
    parent = os.path.dirname(abs_path) or "."
    checks = {
        "db_path": db_path,
        "abs_path": abs_path,
        "cwd": os.getcwd(),
        "parent": parent,
        "parent_exists": os.path.isdir(parent),
        "parent_writable": os.access(parent, os.W_OK) if os.path.isdir(parent) else False,
        "db_exists": os.path.exists(abs_path),
        "db_writable": os.access(abs_path, os.W_OK) if os.path.exists(abs_path) else None,
    }
    for suffix in ("-wal", "-shm"):
        sidecar = abs_path + suffix
        checks[f"{suffix}_exists"] = os.path.exists(sidecar)
        checks[f"{suffix}_writable"] = (
            os.access(sidecar, os.W_OK) if os.path.exists(sidecar) else None
        )
    return ", ".join(f"{key}={value}" for key, value in checks.items())


def _ensure_sqlite_parent_writable(db_path: str) -> None:
    if db_path == ":memory:":
        return

    parent = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(parent, exist_ok=True)
    probe = os.path.join(parent, ".sqlite_write_probe")
    try:
        with open(probe, "a", encoding="utf-8"):
            pass
    except OSError as e:
        raise RuntimeError(
            f"SQLite数据库目录不可写: {e}; {_sqlite_path_diagnostics(db_path)}"
        ) from e
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


# ─────────────────────────────────────────────
# SQLite -- 决策日志 / 自选股 / 持仓 / 复盘
# ─────────────────────────────────────────────

class DecisionLog:
    def __init__(self, db_path: str = None):
        db_path = db_path or DB_PATH
        _ensure_sqlite_parent_writable(db_path)
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        except sqlite3.Error as e:
            raise RuntimeError(
                f"SQLite数据库连接失败: {e}; {_sqlite_path_diagnostics(db_path)}"
            ) from e
        self.conn.row_factory = sqlite3.Row
        try:
            self._init_schema()
        except sqlite3.Error as e:
            self.conn.close()
            raise RuntimeError(
                f"SQLite数据库初始化失败: {e}; {_sqlite_path_diagnostics(db_path)}"
            ) from e

    def _init_schema(self):
        _enable_wal_safely(self.conn, "DecisionLog")
        self.conn.executescript("""
            -- 投资决策记录（append-only，不允许 UPDATE）
            -- 业务含义：历史判断必须可追溯，后续修正通过 retrospectives 追加，不覆盖原判断。
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
            -- 业务含义：这是“等待价格/估值触发”的观察池，通常不代表已买入。
            CREATE TABLE IF NOT EXISTS watchlist (
                stock_code   TEXT PRIMARY KEY,
                stock_name   TEXT NOT NULL,
                reason       TEXT,
                alert_yield  REAL,
                alert_pe_pct REAL,
                alert_price_below REAL,
                watch_price_below REAL,
                alert_note  TEXT,
                added_at     TEXT NOT NULL
            );

            -- 持仓表
            -- 业务含义：这是当前组合事实，用于每轮回答时约束仓位、风险和重复推荐。
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
            -- 业务含义：把“当时为什么这么想”和“后来验证如何”分开保存，方便反思偏差。
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
            -- 业务含义：只沉淀可复用的投资洞察摘要，而不是保存完整聊天噪声。
            CREATE TABLE IF NOT EXISTS episodic_docs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      TEXT UNIQUE NOT NULL,
                text        TEXT NOT NULL,
                metadata    TEXT,
                created_at  TEXT NOT NULL
            );

            -- 每日扫描结果
            -- 业务含义：保存每日自动体检快照，后续可以追踪提醒是否连续出现。
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
        self._ensure_watchlist_columns()
        self.conn.commit()

    def _ensure_watchlist_columns(self):
        existing = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(watchlist)").fetchall()
        }
        migrations = {
            "alert_price_below": "ALTER TABLE watchlist ADD COLUMN alert_price_below REAL",
            "watch_price_below": "ALTER TABLE watchlist ADD COLUMN watch_price_below REAL",
            "alert_note": "ALTER TABLE watchlist ADD COLUMN alert_note TEXT",
        }
        for column, sql in migrations.items():
            if column not in existing:
                self.conn.execute(sql)

    # ── 读操作（Agent 可自主调用）────────────────

    def search_decisions(self, stock_code: str = None,
                         keyword: str = None, limit: int = 10) -> list[dict]:
        # 决策日志检索服务于复盘：既可以按股票代码查，也可以按理由/标签里的主题查。
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
        # 按仓位从高到低返回，让 prompt 里最重要的风险暴露排在前面。
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
        # 决策记录只新增不更新，避免事后改写历史判断。
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
        # 复盘是对原决策的补充证据，保留 outcome 和 missed points 供以后纠偏。
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
        # 持仓是当前状态表，允许 upsert，因为成本、股数、仓位会随交易变化。
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
        # 自选股允许反复更新关注原因和提醒阈值，表示观察条件的演化。
        self.conn.execute("""
            INSERT INTO watchlist
            (stock_code, stock_name, reason, alert_yield, alert_pe_pct,
             alert_price_below, watch_price_below, alert_note, added_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(stock_code) DO UPDATE SET
                stock_name          = excluded.stock_name,
                reason              = excluded.reason,
                alert_yield         = excluded.alert_yield,
                alert_pe_pct        = excluded.alert_pe_pct,
                alert_price_below   = excluded.alert_price_below,
                watch_price_below   = excluded.watch_price_below,
                alert_note          = excluded.alert_note,
                added_at            = excluded.added_at
        """, (
            data["stock_code"],
            data["stock_name"],
            data.get("reason"),
            data.get("alert_yield"),
            data.get("alert_pe_pct"),
            data.get("alert_price_below"),
            data.get("watch_price_below"),
            data.get("alert_note"),
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

    _cross_encoder = None
    _cross_encoder_error: str | None = None

    def __init__(self, db_path: str = None, persist_dir: str = None):
        import chromadb  # pyright: ignore[reportMissingImports]

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
        # 向量库负责“意思相近也能找回”，例如用户换一种说法问同一类投资判断。
        with console_timer("记忆写入", "ChromaDB add"):
            self.collection.add(
                documents=[text],
                metadatas=[meta],
                ids=[doc_id],
            )

        # 2. 写入 SQLite episodic_docs 表
        # SQLite 保存原文和元数据，作为向量命中后的可审计文本来源。
        with console_timer("记忆写入", "SQLite episodic_docs"):
            self.conn.execute("""
                INSERT INTO episodic_docs (doc_id, text, metadata, created_at)
                VALUES (?, ?, ?, ?)
            """, (doc_id, text, json.dumps(meta, ensure_ascii=False), now))

        # 3. 更新 FTS5 索引（存分词后的文本，便于中文检索）
        # FTS5 弥补向量检索对股票代码、数字阈值、精确术语不敏感的问题。
        rowid = self.conn.execute(
            "SELECT id FROM episodic_docs WHERE doc_id=?", (doc_id,)
        ).fetchone()[0]
        with console_timer("记忆写入", "FTS5 index"):
            self.conn.execute(
                "INSERT INTO episodic_fts(rowid, text, doc_id) VALUES (?, ?, ?)",
                (rowid, self._tokenize(text), doc_id),
            )
            self.conn.commit()
        return doc_id

    # ── RRF 融合（纯数学，可独立单测）─────────

    @staticmethod
    def _rrf_fuse(vec_ids: list[str], fts_ids: list[str],
                  k: int = 60, top_k: int = 4) -> list[str]:
        """Reciprocal Rank Fusion: 同一 doc_id 在两路中名次越靠前得分越高。"""
        scores = EpisodicMemory._rrf_score_map(vec_ids, fts_ids, k=k)
        return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    @staticmethod
    def _rrf_score_map(vec_ids: list[str], fts_ids: list[str],
                       k: int = 60) -> dict[str, float]:
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(vec_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(fts_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return scores

    @staticmethod
    def _token_set(text: str) -> set[str]:
        import re

        tokenized = EpisodicMemory._tokenize(text or "").lower()
        tokens = {t for t in tokenized.split() if t}
        tokens.update(re.findall(r"[a-z0-9]+", (text or "").lower()))
        return tokens

    @staticmethod
    def _rerank_candidates(query: str, candidates: list[dict],
                           top_k: int = 4) -> list[dict]:
        # RRF 先做稳健召回，再 rerank 精排；
        # 默认轻量词面排序，只有显式打开 RERANK_ENABLED 才加载重模型。
        if os.environ.get("RERANK_ENABLED", "").lower() in ("1", "true", "yes", "on"):
            return EpisodicMemory._model_rerank_candidates(query, candidates, top_k)
        return EpisodicMemory._lexical_rerank_candidates(query, candidates, top_k)

    @staticmethod
    def _load_cross_encoder():
        if EpisodicMemory._cross_encoder is not None:
            return EpisodicMemory._cross_encoder
        if EpisodicMemory._cross_encoder_error:
            return None

        try:
            from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]

            with console_timer("记忆检索", f"加载rerank模型 {RERANK_MODEL}"):
                EpisodicMemory._cross_encoder = CrossEncoder(RERANK_MODEL)
            return EpisodicMemory._cross_encoder
        except Exception as e:
            EpisodicMemory._cross_encoder_error = str(e)
            print(f"  [rerank] 模型加载失败，降级到轻量rerank: {e}", flush=True)
            return None

    @staticmethod
    def _model_rerank_candidates(query: str, candidates: list[dict],
                                 top_k: int = 4) -> list[dict]:
        model = EpisodicMemory._load_cross_encoder()
        if model is None:
            return EpisodicMemory._lexical_rerank_candidates(query, candidates, top_k)
        if not candidates:
            return []

        try:
            pairs = [(query or "", c.get("text") or "") for c in candidates]
            with console_timer("记忆检索", f"模型rerank model={RERANK_MODEL} pool={len(candidates)} top_k={top_k}"):
                model_scores = model.predict(pairs)

            reranked = []
            for idx, (candidate, score) in enumerate(zip(candidates, model_scores)):
                reranked.append({
                    **candidate,
                    "rerank_score": round(float(score), 6),
                    "rerank_provider": "model",
                    "rerank_model": RERANK_MODEL,
                    "_original_rank": idx,
                })

            reranked.sort(
                key=lambda x: (x["rerank_score"], x["rrf_score"], -x["_original_rank"]),
                reverse=True,
            )
            return [
                {k: v for k, v in item.items() if k != "_original_rank"}
                for item in reranked[:top_k]
            ]
        except Exception as e:
            print(f"  [rerank] 模型推理失败，降级到轻量rerank: {e}", flush=True)
            return EpisodicMemory._lexical_rerank_candidates(query, candidates, top_k)

    @staticmethod
    def _lexical_rerank_candidates(query: str, candidates: list[dict],
                                   top_k: int = 4) -> list[dict]:
        """按查询词与候选文本的直接重合度精排 RRF 结果。

        RRF 仍是主先验：当词面证据不足时，向量+FTS 的原始融合排序保持稳定。
        这让默认检索结果可重复，也避免 Docker 运行时强依赖额外 rerank 模型。
        """
        query_tokens = EpisodicMemory._token_set(query)
        query_lower = (query or "").strip().lower()
        if not candidates:
            return []

        max_rrf = max((c["rrf_score"] for c in candidates), default=0.0) or 1.0
        reranked = []
        for idx, c in enumerate(candidates):
            text = c.get("text") or ""
            doc_tokens = EpisodicMemory._token_set(text)
            if query_tokens:
                lexical_score = len(query_tokens & doc_tokens) / len(query_tokens)
            else:
                lexical_score = 0.0
            if query_lower and query_lower in text.lower():
                lexical_score = min(1.0, lexical_score + 0.25)

            # 70% 保留混合召回排序，30% 奖励词面命中；
            # 这个权重偏保守，防止短关键词把语义相关结果挤掉。
            rrf_score = c["rrf_score"] / max_rrf
            rerank_score = 0.70 * rrf_score + 0.30 * lexical_score
            reranked.append({
                **c,
                "rerank_score": round(rerank_score, 6),
                "rerank_provider": "lexical",
                "_original_rank": idx,
            })

        reranked.sort(
            key=lambda x: (x["rerank_score"], x["rrf_score"], -x["_original_rank"]),
            reverse=True,
        )
        return [
            {k: v for k, v in item.items() if k != "_original_rank"}
            for item in reranked[:top_k]
        ]

    # ── 混合检索主入口 ──────────────────────────

    def retrieve(self, query: str, n_results: int = 8, top_k: int = 4) -> list[dict]:
        with console_timer("记忆检索", "ChromaDB count"):
            collection_count = self.collection.count()
        if collection_count == 0:
            return []

        n = min(n_results, collection_count)

        # ── 路1：ChromaDB 向量检索（HNSW 近似最近邻，语义感知）──
        with console_timer("记忆检索", f"向量库 query n={n}"):
            vec_result = self.collection.query(query_texts=[query], n_results=n)
        vec_ids: list[str] = vec_result["ids"][0]

        # ── 路2：SQLite FTS5 全文检索（精确词匹配，对股票代码/数字敏感）──
        fts_ids: list[str] = []
        with console_timer("记忆检索", "FTS5 query"):
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
        with console_timer("记忆检索", f"RRF fuse vec={len(vec_ids)} fts={len(fts_ids)}"):
            scores = self._rrf_score_map(vec_ids, fts_ids, k=60)
            # 先保留一个比 top_k 更大的候选池，再做 rerank，避免过早丢掉有用记忆。
            rerank_pool_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[
                :min(len(scores), max(RERANK_POOL_SIZE, top_k))
            ]
        if not rerank_pool_ids:
            return []

        with console_timer("记忆检索", f"ChromaDB get rerank_pool={len(rerank_pool_ids)}"):
            fetched = self.collection.get(ids=rerank_pool_ids, include=["documents", "metadatas"])
        fetched_map = {
            id_: (doc, meta)
            for id_, doc, meta in zip(
                fetched.get("ids", []),
                fetched.get("documents", []),
                fetched.get("metadatas", []),
            )
        }

        candidates = [
            {
                "text": fetched_map[id_][0],
                "metadata": fetched_map[id_][1],
                "rrf_score": round(scores[id_], 6),
            }
            for id_ in rerank_pool_ids
            if id_ in fetched_map
        ]

        with console_timer("记忆检索", f"rerank pool={len(candidates)} top_k={top_k}"):
            return self._rerank_candidates(query, candidates, top_k=top_k)


# ─────────────────────────────────────────────
# MemoryManager -- 统一入口
# ─────────────────────────────────────────────

class MemoryManager:
    def __init__(self):
        self.decisions = DecisionLog()
        self.episodic = EpisodicMemory()

    def retrieve_context(self, user_query: str) -> str:
        # 给 Agent 的上下文由两类历史组成：
        # 1. episodic 洞察：适合找相似讨论和用户偏好；
        # 2. decisions 日志：适合追溯明确保存过的投资判断。
        with console_timer("上下文构建", "episodic hybrid retrieve"):
            fragments = self.episodic.retrieve(user_query, n_results=8, top_k=4)
        with console_timer("上下文构建", "SQLite decisions search"):
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
