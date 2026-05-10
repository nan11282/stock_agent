import sqlite3

from memory import EpisodicMemory


class FakeCollection:
    def __init__(self):
        self._ids = ["doc_b", "doc_a", "doc_c"]

    def count(self):
        return len(self._ids)

    def query(self, query_texts, n_results):
        return {"ids": [["doc_b", "doc_a"]]}

    def get(self, ids, include=None):
        docs = {
            "doc_a": ("text-a", {"source": "a"}),
            "doc_b": ("text-b", {"source": "b"}),
        }
        return {
            "ids": ids,
            "documents": [docs[i][0] for i in ids],
            "metadatas": [docs[i][1] for i in ids],
        }


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
