import math

from eval.report import aggregate_pdf


def _approx(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def test_aggregate_counts_pages_by_status():
    pages = [
        {"status": "ok", "text": {"skipped": False}, "table": {"skipped": True},
         "baseline_text_len": 10, "md_text_len": 10, "text_edit_ratio": 0.0,
         "baseline_tables": 0, "md_tables": 0},
        {"status": "scanned", "text": {"skipped": True}, "table": {"skipped": True},
         "baseline_text_len": 0, "md_text_len": 5, "text_edit_ratio": None,
         "baseline_tables": 0, "md_tables": 0},
        {"status": "table_missing", "text": {"skipped": False}, "table": {"skipped": True},
         "baseline_text_len": 20, "md_text_len": 20, "text_edit_ratio": 0.0,
         "baseline_tables": 0, "md_tables": 1},
    ]
    agg = aggregate_pdf(pages, lang="zh")
    assert agg["pages_total"] == 3
    assert agg["pages_scanned"] == 1
    assert agg["pages_table_missing"] == 1


def test_aggregate_text_micro_average():
    pages = [
        {"status": "ok", "text": {"skipped": False}, "table": {"skipped": True},
         "baseline_text_len": 10, "md_text_len": 10, "text_edit_ratio": 0.2,
         "baseline_tables": 0, "md_tables": 0,
         "_text_tp": 8, "_text_fp": 2, "_text_fn": 2},
        {"status": "ok", "text": {"skipped": False}, "table": {"skipped": True},
         "baseline_text_len": 10, "md_text_len": 10, "text_edit_ratio": 0.4,
         "baseline_tables": 0, "md_tables": 0,
         "_text_tp": 6, "_text_fp": 4, "_text_fn": 4},
    ]
    agg = aggregate_pdf(pages, lang="zh")
    assert _approx(agg["text_P"], 0.7)
    assert _approx(agg["text_R"], 0.7)


def test_aggregate_table_micro_average():
    pages = [
        {"status": "ok", "text": {"skipped": True},
         "table": {"skipped": False, "tp": 5, "fp": 1, "fn": 2,
                   "tables_matched": 1, "tables_total_baseline": 1},
         "baseline_text_len": 0, "md_text_len": 0, "text_edit_ratio": None,
         "baseline_tables": 1, "md_tables": 1},
        {"status": "ok", "text": {"skipped": True},
         "table": {"skipped": False, "tp": 3, "fp": 2, "fn": 1,
                   "tables_matched": 1, "tables_total_baseline": 2},
         "baseline_text_len": 0, "md_text_len": 0, "text_edit_ratio": None,
         "baseline_tables": 2, "md_tables": 1},
    ]
    agg = aggregate_pdf(pages, lang="zh")
    assert _approx(agg["table_P"], 8 / 11)
    assert _approx(agg["table_R"], 8 / 11)
    assert agg["tables_total"] == 3
    assert agg["tables_matched"] == 2
