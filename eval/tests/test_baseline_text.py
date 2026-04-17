from pathlib import Path

import pytest

from eval.baseline import extract_baseline_text

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.pdf"


def test_extract_text_returns_one_string_per_page():
    texts = extract_baseline_text(FIXTURE)
    assert len(texts) == 2
    assert "hello world" in texts[0]
    assert "page two content" in texts[1]


def test_extract_text_applies_normalization():
    texts = extract_baseline_text(FIXTURE)
    assert texts[0] == texts[0].lower()
    assert "  " not in texts[0]


def test_extract_text_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_baseline_text(Path("/tmp/nonexistent_xyz.pdf"))
