"""端到端冒烟：只跑 1 份最短的 PDF（昆山峰瑞 11 页），验证 4 种产物非空。

如果 config.yaml 不存在或对应 PDF/md_dir 缺失，跳过。
"""
from pathlib import Path

import pytest
import yaml

from eval.run_eval import run_evaluation

CONFIG_PATH = Path(__file__).resolve().parents[2] / "eval" / "config.yaml"


def _find_kunshan_item():
    if not CONFIG_PATH.exists():
        return None
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for it in cfg.get("items", []):
        if "昆山峰瑞" in it["name"]:
            if Path(it["pdf"]).exists() and Path(it["md_dir"]).exists():
                return it
    return None


@pytest.mark.skipif(_find_kunshan_item() is None, reason="config.yaml or kunshan PDF not available")
def test_smoke_kunshan(tmp_path):
    item = _find_kunshan_item()
    results = run_evaluation(items=[item], output_dir=tmp_path)
    assert len(results) == 1
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.md").exists()
    sub = tmp_path / item["name"]
    assert (sub / "per_page.csv").exists()
    assert (sub / "raw.json").exists()
    assert (sub / "bad_cases.md").exists()
    import csv
    rows = list(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["pdf"] == item["name"]
