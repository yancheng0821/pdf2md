from pathlib import Path

from eval.baseline import extract_baseline_tables

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_with_table.pdf"
NO_TABLE = Path(__file__).parent / "fixtures" / "tiny.pdf"


def test_extract_tables_returns_one_list_per_page():
    tables = extract_baseline_tables(FIXTURE)
    assert len(tables) == 1
    assert isinstance(tables[0], list)


def test_extract_tables_finds_table_with_cells():
    tables = extract_baseline_tables(FIXTURE)
    page_tables = tables[0]
    assert len(page_tables) >= 1
    all_cells = [c for t in page_tables for row in t for c in row]
    assert "Alice" in all_cells
    assert "30" in all_cells
    assert "NYC" in all_cells


def test_extract_tables_normalizes_none_to_empty_string():
    tables = extract_baseline_tables(FIXTURE)
    for page in tables:
        for table in page:
            for row in table:
                for cell in row:
                    assert isinstance(cell, str)


def test_extract_tables_no_table_page_returns_empty_list():
    tables = extract_baseline_tables(NO_TABLE)
    assert len(tables) == 2
    for page in tables:
        assert isinstance(page, list)
