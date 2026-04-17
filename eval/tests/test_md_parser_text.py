from eval.md_parser import strip_markdown


def test_strip_headings():
    assert strip_markdown("# Title\n## Subtitle\ncontent") == "title subtitle content"


def test_strip_emphasis():
    assert strip_markdown("**bold** and *italic* text") == "bold and italic text"


def test_strip_code_fences():
    md = "before\n```python\nx = 1\n```\nafter"
    assert "```" not in strip_markdown(md)
    assert "python" not in strip_markdown(md)
    assert "x = 1" not in strip_markdown(md)


def test_strip_inline_code():
    assert strip_markdown("use `foo()` now") == "use now"


def test_strip_links_and_images():
    assert strip_markdown("see [here](http://x) now") == "see here now"
    assert strip_markdown("![alt](img.png) caption") == "caption"


def test_strip_list_markers():
    assert strip_markdown("- item a\n- item b\n1. numbered") == "item a item b numbered"


def test_strip_blockquotes():
    assert strip_markdown("> quoted\n> line two") == "quoted line two"


def test_strip_tables_removed():
    md = "before\n| a | b |\n|---|---|\n| 1 | 2 |\nafter"
    result = strip_markdown(md)
    assert "|" not in result
    assert "---" not in result


def test_strip_preserves_cjk():
    md = "# 标题\n**基金**成立于2020年"
    result = strip_markdown(md)
    assert "标题" in result
    assert "基金" in result
    assert "2020" in result


def test_strip_empty_string():
    assert strip_markdown("") == ""
