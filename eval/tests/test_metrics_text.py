import math

from eval.metrics_text import (
    char_metrics,
    edit_distance,
    edit_ratio,
)


def _approx(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def test_char_metrics_identical():
    m = char_metrics("hello world", "hello world")
    assert _approx(m["precision"], 1.0)
    assert _approx(m["recall"], 1.0)
    assert _approx(m["f1"], 1.0)


def test_char_metrics_subset_md_is_substring_of_baseline():
    m = char_metrics(baseline="hello world", md="hello")
    assert _approx(m["precision"], 1.0)
    assert m["recall"] < 1.0
    assert m["recall"] > 0.0


def test_char_metrics_superset_md_has_extra():
    m = char_metrics(baseline="hello", md="hello world")
    assert m["precision"] < 1.0
    assert _approx(m["recall"], 1.0)


def test_char_metrics_completely_different():
    m = char_metrics("abcdef", "xyzuvw")
    assert m["f1"] < 0.2


def test_char_metrics_empty_md_gives_zero_recall():
    m = char_metrics(baseline="hello", md="")
    assert _approx(m["precision"], 0.0)
    assert _approx(m["recall"], 0.0)
    assert _approx(m["f1"], 0.0)


def test_char_metrics_both_empty_returns_skipped():
    m = char_metrics(baseline="", md="")
    assert m.get("skipped") is True


def test_edit_distance_identical():
    assert edit_distance("abc", "abc") == 0


def test_edit_distance_insertions():
    assert edit_distance("abc", "abxc") == 1


def test_edit_distance_deletions_and_substitutions():
    assert edit_distance("kitten", "sitting") == 3


def test_edit_ratio_normalized_by_baseline():
    assert _approx(edit_ratio("abc", "xyz"), 1.0)
    assert _approx(edit_ratio("abc", "abc"), 0.0)


def test_edit_ratio_empty_baseline_returns_zero():
    assert edit_ratio("", "anything") == 0.0


from eval.metrics_text import word_metrics


def test_word_metrics_identical():
    m = word_metrics("hello world foo bar", "hello world foo bar")
    assert _approx(m["precision"], 1.0)
    assert _approx(m["recall"], 1.0)
    assert _approx(m["f1"], 1.0)


def test_word_metrics_whitespace_tokenized():
    m = word_metrics("a b c", "a  b\nc")
    assert _approx(m["f1"], 1.0)


def test_word_metrics_partial_overlap():
    m = word_metrics(baseline="the quick brown fox", md="the lazy fox")
    assert 0 < m["precision"] < 1
    assert 0 < m["recall"] < 1


def test_word_metrics_empty_both_skipped():
    m = word_metrics("", "")
    assert m.get("skipped") is True
