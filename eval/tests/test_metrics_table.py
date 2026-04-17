import math

from eval.metrics_table import cell_multiset, pair_metrics


def _approx(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def test_cell_multiset_flattens_2d_array():
    t = [["a", "b"], ["c", "a"]]
    ms = cell_multiset(t)
    assert ms["a"] == 2
    assert ms["b"] == 1
    assert ms["c"] == 1


def test_cell_multiset_skips_empty_cells():
    t = [["a", ""], ["", "b"]]
    ms = cell_multiset(t)
    assert "" not in ms
    assert ms["a"] == 1
    assert ms["b"] == 1


def test_pair_metrics_identical_tables():
    t = [["a", "b"], ["1", "2"]]
    m = pair_metrics(t, t)
    assert _approx(m["precision"], 1.0)
    assert _approx(m["recall"], 1.0)
    assert _approx(m["f1"], 1.0)


def test_pair_metrics_md_missing_row():
    baseline = [["a", "b"], ["1", "2"], ["3", "4"]]
    md = [["a", "b"], ["1", "2"]]
    m = pair_metrics(md=md, baseline=baseline)
    assert _approx(m["precision"], 1.0)
    assert _approx(m["recall"], 4 / 6)


def test_pair_metrics_md_extra_column():
    baseline = [["a", "b"], ["1", "2"]]
    md = [["a", "b", "EXTRA"], ["1", "2", "EXTRA2"]]
    m = pair_metrics(md=md, baseline=baseline)
    assert _approx(m["precision"], 4 / 6)
    assert _approx(m["recall"], 1.0)


def test_pair_metrics_disjoint_tables():
    m = pair_metrics(md=[["x"]], baseline=[["y"]])
    assert _approx(m["precision"], 0.0)
    assert _approx(m["recall"], 0.0)
    assert _approx(m["f1"], 0.0)
