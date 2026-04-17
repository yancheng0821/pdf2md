from eval.report import evaluate_page


def test_evaluate_page_both_text_and_tables():
    r = evaluate_page(
        baseline_text="hello world this is a test",
        md_text="hello world this is a test",
        baseline_tables=[[["a", "b"], ["1", "2"]]],
        md_tables=[[["a", "b"], ["1", "2"]]],
        lang="en",
    )
    assert r["status"] == "ok"
    assert r["text"]["f1"] == 1.0
    assert r["table"]["f1"] == 1.0
    assert "word" in r
    assert r["word"]["f1"] == 1.0


def test_evaluate_page_scanned():
    r = evaluate_page(
        baseline_text="",
        md_text="ocr reconstructed text here long enough",
        baseline_tables=[],
        md_tables=[],
        lang="zh",
    )
    assert r["status"] == "scanned"
    assert r["text"].get("skipped") is True


def test_evaluate_page_table_baseline_missing():
    r = evaluate_page(
        baseline_text="some baseline text long enough",
        md_text="some baseline text long enough",
        baseline_tables=[],
        md_tables=[[["x", "y"]]],
        lang="zh",
    )
    assert r["status"] == "table_missing"
    assert r["text"].get("skipped") is False
    assert r["table"].get("skipped") is True


def test_evaluate_page_chinese_no_word_metrics():
    r = evaluate_page(
        baseline_text="中文内容测试内容",
        md_text="中文内容测试内容",
        baseline_tables=[],
        md_tables=[],
        lang="zh",
    )
    assert "word" not in r or r.get("word") is None
