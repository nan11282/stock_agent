"""RRF 融合纯数学测试：不依赖 ChromaDB / SQLite。"""

from memory import EpisodicMemory


def test_single_path_only_vec():
    out = EpisodicMemory._rrf_fuse(["a", "b", "c"], [], k=60, top_k=4)
    assert out == ["a", "b", "c"]


def test_single_path_only_fts():
    out = EpisodicMemory._rrf_fuse([], ["x", "y"], k=60, top_k=4)
    assert out == ["x", "y"]


def test_doc_in_both_paths_ranks_higher():
    # 'common' 在两路都靠前 → 总分应超过任一路独占的项
    vec = ["common", "vec_only", "z"]
    fts = ["common", "fts_only", "z"]
    out = EpisodicMemory._rrf_fuse(vec, fts, k=60, top_k=4)
    assert out[0] == "common"


def test_top_k_truncation():
    vec = [f"v{i}" for i in range(10)]
    fts = [f"f{i}" for i in range(10)]
    out = EpisodicMemory._rrf_fuse(vec, fts, k=60, top_k=3)
    assert len(out) == 3


def test_score_formula_first_rank_beats_second():
    # 仅一路、两个 doc，名次第 0 应排在第 1 之前
    out = EpisodicMemory._rrf_fuse(["first", "second"], [], k=60, top_k=2)
    assert out == ["first", "second"]


def test_k_smoothing_effect():
    # K 越大，rank 之间差距越小
    # K=1 时第 0 名得分 1/2，第 1 名 1/3 → 差 0.166
    # K=1000 时差 ≈ 1/1001 - 1/1002 ≈ 1e-6
    # 这里只验证排序仍然稳定，不验证数值
    out_small_k = EpisodicMemory._rrf_fuse(["a", "b"], ["b", "a"], k=1, top_k=2)
    out_large_k = EpisodicMemory._rrf_fuse(["a", "b"], ["b", "a"], k=1000, top_k=2)
    # 两路各拥护一个，平局 → 排序由插入顺序/dict 顺序决定，不严格断言
    assert set(out_small_k) == {"a", "b"}
    assert set(out_large_k) == {"a", "b"}


def test_empty_inputs():
    assert EpisodicMemory._rrf_fuse([], [], k=60, top_k=4) == []
