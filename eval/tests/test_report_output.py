import csv
import json

from eval.report import write_pdf_report, write_summary


def _fake_pdf_result(name, lang="zh"):
    return {
        "name": name,
        "lang": lang,
        "agg": {
            "lang": lang,
            "pages_total": 3,
            "pages_scanned": 0,
            "pages_table_missing": 0,
            "pages_missing": 0,
            "text_P": 0.9, "text_R": 0.85, "text_F1": 0.87,
            "text_edit_ratio": 0.1,
            "table_P": 0.8, "table_R": 0.7, "table_F1": 0.75,
            "tables_total": 5, "tables_matched": 4,
            "text_P_word": None, "text_R_word": None, "text_F1_word": None,
        },
        "pages": [
            {"page": 1, "status": "ok", "baseline_text_len": 100, "md_text_len": 98,
             "text": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "skipped": False},
             "text_edit_ratio": 0.05, "baseline_tables": 1, "md_tables": 1,
             "table": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "skipped": False},
             "md_method": "native"},
        ],
    }


def test_write_pdf_report_creates_files(tmp_path):
    result = _fake_pdf_result("sample")
    write_pdf_report(result, tmp_path)
    sub = tmp_path / "sample"
    assert (sub / "per_page.csv").exists()
    assert (sub / "raw.json").exists()
    rows = list(csv.DictReader((sub / "per_page.csv").open(encoding="utf-8")))
    assert rows[0]["page"] == "1"
    assert rows[0]["status"] == "ok"
    data = json.loads((sub / "raw.json").read_text(encoding="utf-8"))
    assert data["name"] == "sample"
    assert data["agg"]["pages_total"] == 3


def test_write_summary_writes_csv_and_md(tmp_path):
    results = [_fake_pdf_result("a"), _fake_pdf_result("b", lang="en")]
    results[1]["agg"]["text_P_word"] = 0.95
    results[1]["agg"]["text_R_word"] = 0.9
    results[1]["agg"]["text_F1_word"] = 0.92
    write_summary(results, tmp_path)
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.md").exists()
    rows = list(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    zh = [r for r in rows if r["pdf"] == "a"][0]
    assert zh["text_P_word"] == ""
    md_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "TL;DR" in md_text or "tl;dr" in md_text.lower()
