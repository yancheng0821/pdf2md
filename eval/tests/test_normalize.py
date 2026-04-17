from eval.normalize import normalize_text


def test_normalize_nfkc_fullwidth_to_halfwidth():
    assert normalize_text("ＡＢＣ１２３") == "abc123"


def test_normalize_removes_zero_width():
    assert normalize_text("hel\u200blo\ufefff") == "hellof"


def test_normalize_collapses_whitespace():
    assert normalize_text("a   b\n\n\nc\t\td") == "a b c d"


def test_normalize_lowercases_ascii():
    assert normalize_text("Hello World") == "hello world"


def test_normalize_preserves_cjk():
    assert normalize_text("本基金  成立于\u200b2020年") == "本基金 成立于2020年"


def test_normalize_empty_string():
    assert normalize_text("") == ""


def test_normalize_none_safe():
    assert normalize_text(None) == ""


def test_normalize_removes_thousands_separator():
    """1,100.00 和 1100.00 应该归一化成相同字符串，避免千分位差异被当成错。"""
    assert normalize_text("1,100.00") == normalize_text("1100.00")
    assert normalize_text("1,154,473,456.44") == normalize_text("1154473456.44")


def test_normalize_preserves_list_commas():
    """数字间的逗号才去掉；句中的逗号（两侧非数字）保留。"""
    # 英文句子里的逗号（字母后的逗号）不去掉
    result = normalize_text("alice, bob, and 1,100 dollars")
    assert ", bob" in result
    assert "1100" in result  # 数字里的千分位去掉
    # 中文标点 NFKC 会归为半角，但逗号本身仍保留（只是从 "，" 变成 ","）
    result_cn = normalize_text("第一，第二")
    assert "," in result_cn  # 保留了（虽然从全角归成了半角）
    assert "第一,第二" == result_cn
