"""表格指标：多表贪心 Jaccard 配对 + 单元格 multiset P/R/F1。

不按行列位置对齐，而是把单元格拉平成 multiset 后求交集。代价是放过"行错位但内容全在"
的情况，但能容忍 pdfplumber 常见的抽多/抽少一行表头。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

Table = list[list[str]]


def cell_multiset(table: Table) -> Counter:
    """把 2D 表拉平成 {单元格值: 出现次数} 的 Counter；空字符串跳过。"""
    c: Counter = Counter()
    for row in table:
        for cell in row:
            if cell:
                c[cell] += 1
    return c


def _prf_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def pair_metrics(md: Table, baseline: Table) -> dict[str, Any]:
    """对一对已配对的 (md_table, baseline_table) 计算 P/R/F1。"""
    ms_md = cell_multiset(md)
    ms_base = cell_multiset(baseline)
    total_md = sum(ms_md.values())
    total_base = sum(ms_base.values())
    tp = sum((ms_md & ms_base).values())
    fp = total_md - tp
    fn = total_base - tp
    result = _prf_from_counts(tp, fp, fn)
    result["tp"] = tp
    result["fp"] = fp
    result["fn"] = fn
    return result
