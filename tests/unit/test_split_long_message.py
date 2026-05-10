from telegram_bot import _split_long_message


def test_short_message_returns_one_part():
    parts = _split_long_message("hello world", limit=4000)
    assert parts == ["hello world"]


def test_long_message_splits_by_paragraph():
    para_a = "a" * 1500
    para_b = "b" * 1500
    para_c = "c" * 1500
    msg = "\n".join([para_a, para_b, para_c])
    parts = _split_long_message(msg, limit=2000)
    assert len(parts) >= 2
    assert all(len(p) <= 2000 for p in parts)
    # 顺序保留
    joined = "".join(parts)
    assert para_a in joined and para_b in joined and para_c in joined


def test_no_part_exceeds_limit_even_under_pressure():
    msg = "\n".join(["x" * 100] * 60)  # 6000+ 字符
    parts = _split_long_message(msg, limit=500)
    for p in parts:
        assert len(p) <= 500


def test_single_long_paragraph_is_chunked():
    msg = "x" * 1200
    parts = _split_long_message(msg, limit=500)
    assert len(parts) == 3
    assert all(len(p) <= 500 for p in parts)


def test_empty_input_returns_empty_or_safe():
    parts = _split_long_message("", limit=4000)
    # 当前实现返回 [""[:4000]] = [""]，不应崩溃
    assert isinstance(parts, list)
