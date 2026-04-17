from eval.md_parser import parse_md_tables, parse_page, read_page_md


def test_parse_single_markdown_table():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    tables = parse_md_tables(md)
    assert len(tables) == 1
    assert tables[0] == [["a", "b"], ["1", "2"], ["3", "4"]]


def test_parse_two_markdown_tables():
    md = (
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "text between\n\n"
        "| x | y |\n|---|---|\n| 9 | 8 |\n"
    )
    tables = parse_md_tables(md)
    assert len(tables) == 2
    assert tables[0][0] == ["a", "b"]
    assert tables[1][0] == ["x", "y"]


def test_parse_strips_cell_whitespace():
    md = "|  a  |  b  |\n|---|---|\n|   1   |   2   |"
    tables = parse_md_tables(md)
    assert tables[0] == [["a", "b"], ["1", "2"]]


def test_parse_handles_html_table():
    md = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    tables = parse_md_tables(md)
    assert len(tables) == 1
    assert tables[0] == [["a", "b"], ["1", "2"]]


def test_parse_no_tables_returns_empty_list():
    assert parse_md_tables("just plain text") == []
    assert parse_md_tables("") == []


def test_parse_skips_separator_row():
    md = "| a | b |\n|---|---|\n| 1 | 2 |"
    tables = parse_md_tables(md)
    for row in tables[0]:
        for cell in row:
            assert "---" not in cell


def test_read_page_md_returns_content(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "001.md").write_text("# Title\ncontent", encoding="utf-8")
    assert read_page_md(tmp_path, 1) == "# Title\ncontent"


def test_read_page_md_missing_returns_none(tmp_path):
    assert read_page_md(tmp_path, 999) is None


def test_parse_page_splits_text_and_tables(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "001.md").write_text(
        "# Title\nintro text\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
        encoding="utf-8",
    )
    result = parse_page(tmp_path, 1)
    assert result is not None
    text, tables = result
    assert "title" in text
    assert "intro text" in text
    assert len(tables) == 1
    assert tables[0][0] == ["a", "b"]
