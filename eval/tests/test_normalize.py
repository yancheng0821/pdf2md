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
