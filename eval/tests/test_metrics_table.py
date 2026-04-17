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


from eval.metrics_table import page_table_metrics


def test_page_metrics_no_tables_returns_skipped():
    m = page_table_metrics(md_tables=[], baseline_tables=[])
    assert m.get("skipped") is True


def test_page_metrics_baseline_missing_returns_skipped():
    m = page_table_metrics(md_tables=[[["a", "b"]]], baseline_tables=[])
    assert m.get("skipped") is True
    assert m.get("reason") == "baseline_missing"


def test_page_metrics_md_missing_all_tables_fn():
    baseline = [[["a", "b"], ["1", "2"]]]
    m = page_table_metrics(md_tables=[], baseline_tables=baseline)
    assert m.get("skipped") is False
    assert _approx(m["precision"], 0.0)
    assert _approx(m["recall"], 0.0)
    assert m["tables_matched"] == 0


def test_page_metrics_one_to_one_match():
    t = [["a", "b"], ["1", "2"]]
    m = page_table_metrics(md_tables=[t], baseline_tables=[t])
    assert _approx(m["f1"], 1.0)
    assert m["tables_matched"] == 1


def test_page_metrics_greedy_pairing_by_jaccard():
    base1 = [["x", "y"], ["1", "2"]]
    base2 = [["a", "b"], ["9", "8"]]
    md1 = [["a", "b"], ["9", "8"]]
    md2 = [["x", "y"], ["1", "2"]]
    m = page_table_metrics(md_tables=[md1, md2], baseline_tables=[base1, base2])
    assert _approx(m["f1"], 1.0)
    assert m["tables_matched"] == 2


def test_page_metrics_unpaired_md_table_counts_as_fp():
    base = [["a", "b"]]
    md_matched = [["a", "b"]]
    md_unmatched = [["zzz", "qqq"]]
    m = page_table_metrics(md_tables=[md_matched, md_unmatched], baseline_tables=[base])
    assert _approx(m["precision"], 0.5)
    assert _approx(m["recall"], 1.0)
    assert m["tables_matched"] == 1
