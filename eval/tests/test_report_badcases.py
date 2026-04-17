from eval.report import write_bad_cases


def _page(num, status, text_f1, table_f1, md_method="native"):
    return {
        "page": num,
        "status": status,
        "baseline_text_len": 100,
        "md_text_len": 100,
        "text": {"precision": text_f1, "recall": text_f1, "f1": text_f1,
                 "skipped": status == "scanned"},
        "text_edit_ratio": 0.1,
        "baseline_tables": 1,
        "md_tables": 1,
        "table": {"precision": table_f1, "recall": table_f1, "f1": table_f1,
                  "skipped": table_f1 is None},
        "md_method": md_method,
        "_baseline_text": "hello world this is the original baseline text for page " + str(num),
        "_md_text": "hallo wrld ths is the original baseline txt for page " + str(num),
        "_baseline_tables": [[["a", "b"], ["1", "2"]]],
        "_md_tables": [[["a", "b"], ["1", "99"]]],
    }


def test_bad_cases_selects_lowest_f1_ok_pages(tmp_path):
    result = {
        "name": "demo",
        "lang": "zh",
        "pages": [
            _page(1, "ok", 0.9, 0.8),
            _page(2, "ok", 0.3, 0.5),
            _page(3, "ok", 0.5, 0.4),
            _page(4, "ok", 0.7, 0.7),
            _page(5, "scanned", None, None),
        ],
    }
    write_bad_cases(result, tmp_path)
    text = (tmp_path / "demo" / "bad_cases.md").read_text(encoding="utf-8")
    assert "Page 2" in text
    assert "Page 3" in text
    assert "Page 5" not in text
    assert "Text diff" in text or "text diff" in text.lower()


def test_bad_cases_empty_when_all_pages_skipped(tmp_path):
    result = {
        "name": "demo",
        "lang": "zh",
        "pages": [_page(1, "scanned", None, None)],
    }
    write_bad_cases(result, tmp_path)
    text = (tmp_path / "demo" / "bad_cases.md").read_text(encoding="utf-8")
    assert "无可用 bad case" in text or "no bad case" in text.lower()
