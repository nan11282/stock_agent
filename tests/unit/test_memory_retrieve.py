import sqlite3

from memory import EpisodicMemory


class FakeCollection:
    def __init__(self):
        self.docs = {
            "doc_a": ("text-a", {"source": "a"}),
            "doc_b": ("text-b", {"source": "b"}),
            "doc_c": ("text-c", {"source": "c"}),
        }

    def count(self):
        return len(self.docs)

    def query(self, query_texts, n_results):
        return {"ids": [["doc_b", "doc_a"]]}

    def get(self, ids, include=None):
        existing_ids = [doc_id for doc_id in ids if doc_id in self.docs]
        if include == []:
            return {"ids": existing_ids}
        return {
            "ids": existing_ids,
            "documents": [self.docs[i][0] for i in existing_ids],
            "metadatas": [self.docs[i][1] for i in existing_ids],
        }

    def update(self, ids, documents, metadatas):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            self.docs[doc_id] = (doc, meta)

    def add(self, ids, documents, metadatas):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            self.docs[doc_id] = (doc, meta)

    def delete(self, ids):
        for doc_id in ids:
            self.docs.pop(doc_id, None)


def test_retrieve_returns_scored_results():
    mem = EpisodicMemory.__new__(EpisodicMemory)
    mem.conn = sqlite3.connect(":memory:")
    mem.conn.row_factory = sqlite3.Row
    mem.conn.execute("""
        CREATE VIRTUAL TABLE episodic_fts
        USING fts5(text, doc_id UNINDEXED)
    """)
    mem.collection = FakeCollection()

    out = mem.retrieve("测试查询", top_k=2)

    assert len(out) == 2
    assert out[0]["text"] == "text-b"
    assert out[0]["rrf_score"] >= out[1]["rrf_score"]
    assert out[0]["metadata"]["source"] == "b"
    assert "rerank_score" in out[0]


def test_update_insight_syncs_sqlite_fts_and_chroma():
    mem = EpisodicMemory.__new__(EpisodicMemory)
    mem.conn = sqlite3.connect(":memory:")
    mem.conn.row_factory = sqlite3.Row
    mem.conn.executescript("""
        CREATE TABLE episodic_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE episodic_fts
        USING fts5(text, doc_id UNINDEXED);
    """)
    mem.conn.execute(
        """
        INSERT INTO episodic_docs (doc_id, text, metadata, created_at)
        VALUES ('doc_a', '旧文本', '{"source": "manual"}', '2026-05-20T09:00:00')
        """
    )
    mem.conn.execute(
        "INSERT INTO episodic_fts(rowid, text, doc_id) VALUES (1, ?, 'doc_a')",
        (EpisodicMemory._tokenize("旧文本"),),
    )
    mem.conn.commit()
    mem.collection = FakeCollection()

    assert mem.update_insight("doc_a", "中国石化 高股息", {"source": "admin"})

    row = mem.conn.execute(
        "SELECT text, metadata FROM episodic_docs WHERE doc_id='doc_a'"
    ).fetchone()
    fts_row = mem.conn.execute(
        "SELECT text FROM episodic_fts WHERE doc_id='doc_a'"
    ).fetchone()
    chroma_doc, chroma_meta = mem.collection.docs["doc_a"]

    assert row["text"] == "中国石化 高股息"
    assert "updated_at" in row["metadata"]
    assert "中国" in fts_row["text"]
    assert "石化" in fts_row["text"]
    assert chroma_doc == "中国石化 高股息"
    assert chroma_meta["source"] == "admin"


def test_delete_insight_removes_sqlite_fts_and_chroma():
    mem = EpisodicMemory.__new__(EpisodicMemory)
    mem.conn = sqlite3.connect(":memory:")
    mem.conn.row_factory = sqlite3.Row
    mem.conn.executescript("""
        CREATE TABLE episodic_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE episodic_fts
        USING fts5(text, doc_id UNINDEXED);
    """)
    mem.conn.execute(
        """
        INSERT INTO episodic_docs (doc_id, text, metadata, created_at)
        VALUES ('doc_a', '待删除', '{}', '2026-05-20T09:00:00')
        """
    )
    mem.conn.execute(
        "INSERT INTO episodic_fts(rowid, text, doc_id) VALUES (1, ?, 'doc_a')",
        (EpisodicMemory._tokenize("待删除"),),
    )
    mem.conn.commit()
    mem.collection = FakeCollection()

    assert mem.delete_insight("doc_a")

    doc_count = mem.conn.execute("SELECT COUNT(*) FROM episodic_docs").fetchone()[0]
    fts_count = mem.conn.execute("SELECT COUNT(*) FROM episodic_fts").fetchone()[0]
    assert doc_count == 0
    assert fts_count == 0
    assert "doc_a" not in mem.collection.docs


def test_rerank_promotes_query_matching_candidate():
    candidates = [
        {
            "text": "银行 高股息 防御",
            "metadata": {"source": "rrf-first"},
            "rrf_score": 0.020000,
        },
        {
            "text": "中国石化 油价 炼化 分红",
            "metadata": {"source": "query-match"},
            "rrf_score": 0.019900,
        },
    ]

    out = EpisodicMemory._rerank_candidates("中国石化 分红", candidates, top_k=1)

    assert out[0]["metadata"]["source"] == "query-match"


def test_model_rerank_falls_back_to_lexical_when_model_unavailable(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    monkeypatch.setattr(EpisodicMemory, "_cross_encoder", None)
    monkeypatch.setattr(EpisodicMemory, "_cross_encoder_error", "disabled in test")
    candidates = [
        {
            "text": "银行 高股息 防御",
            "metadata": {"source": "rrf-first"},
            "rrf_score": 0.020000,
        },
        {
            "text": "中国石化 油价 炼化 分红",
            "metadata": {"source": "query-match"},
            "rrf_score": 0.019900,
        },
    ]

    out = EpisodicMemory._rerank_candidates("中国石化 分红", candidates, top_k=1)

    assert out[0]["metadata"]["source"] == "query-match"
    assert out[0]["rerank_provider"] == "lexical"
