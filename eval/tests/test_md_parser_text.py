from eval.md_parser import read_page_md, strip_markdown


def test_strip_headings():
    assert strip_markdown("# Title\n## Subtitle\ncontent") == "title subtitle content"


def test_strip_emphasis():
    assert strip_markdown("**bold** and *italic* text") == "bold and italic text"


def test_strip_code_fences():
    md = "before\n```python\nx = 1\n```\nafter"
    s = strip_markdown(md)
    assert "```" not in s and "python" not in s and "x = 1" not in s


def test_strip_inline_code():
    assert strip_markdown("use `foo()` now") == "use now"


def test_strip_links_and_images():
    assert strip_markdown("see [here](http://x) now") == "see here now"
    assert strip_markdown("![alt](img.png) caption") == "caption"


def test_strip_list_markers():
    assert strip_markdown("- item a\n- item b\n1. numbered") == "item a item b numbered"


def test_strip_blockquotes():
    assert strip_markdown("> quoted\n> line two") == "quoted line two"


def test_strip_tables_preserves_cell_content():
    md = "before\n| a | b |\n|---|---|\n| 1 | 2 |\nafter"
    result = strip_markdown(md)
    assert "|" not in result
    assert "---" not in result
    for token in ("before", "a", "b", "1", "2", "after"):
        assert token in result


def test_strip_html_table_preserves_cell_content():
    """HTML table 标签剥掉但单元格文字保留（pdf2md 偶尔输出 HTML table，里面是真实数据）。"""
    md = "before\n<table><tr><td>浦发银行</td><td>苏州支行</td></tr></table>\nafter"
    result = strip_markdown(md)
    assert "<" not in result
    assert "td" not in result
    for token in ("before", "浦发银行", "苏州支行", "after"):
        assert token in result


def test_strip_br_tag_removed_preserve_content():
    md = "一段文字<br>第二段<br/>第三段"
    result = strip_markdown(md)
    assert "<" not in result
    for token in ("一段文字", "第二段", "第三段"):
        assert token in result


def test_strip_html_comment_removed():
    md = "<!-- page:5 method:native -->\n正文内容"
    result = strip_markdown(md)
    assert "page:" not in result
    assert "method:" not in result
    assert "正文内容" in result


def test_strip_preserves_cjk():
    md = "# 标题\n**基金**成立于2020年"
    result = strip_markdown(md)
    assert "标题" in result and "基金" in result and "2020" in result


def test_strip_empty_string():
    assert strip_markdown("") == ""


def test_read_page_md_returns_content(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "001.md").write_text("# Title\ncontent", encoding="utf-8")
    assert read_page_md(tmp_path, 1) == "# Title\ncontent"


def test_read_page_md_missing_returns_none(tmp_path):
    assert read_page_md(tmp_path, 999) is None
