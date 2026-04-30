from memory import EpisodicMemory


def test_tokenize_chinese_phrase():
    out = EpisodicMemory._tokenize("中国石化股息率高")
    tokens = out.split()
    # jieba 应至少能切出"中国石化"或"中国"+"石化"
    assert any("石化" in t or "中国" in t for t in tokens)
    assert "股息" in out or "股息率" in out


def test_tokenize_preserves_stock_code():
    # 股票代码这种数字串 jieba 通常整体保留
    out = EpisodicMemory._tokenize("分析600028这只股票")
    assert "600028" in out.split()


def test_tokenize_strips_whitespace_tokens():
    out = EpisodicMemory._tokenize("a   b   c")
    tokens = out.split()
    # 不应有空 token
    assert all(t.strip() == t and t != "" for t in tokens)


def test_tokenize_empty():
    out = EpisodicMemory._tokenize("")
    assert out == ""
