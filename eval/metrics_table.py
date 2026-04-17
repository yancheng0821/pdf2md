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


JACCARD_THRESHOLD = 0.1


def _jaccard(a: Table, b: Table) -> float:
    sa = set(cell_multiset(a).keys())
    sb = set(cell_multiset(b).keys())
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def page_table_metrics(
    md_tables: list[Table],
    baseline_tables: list[Table],
) -> dict[str, Any]:
    """页级表格指标：贪心配对 + TP/FP/FN 累加。"""
    if not baseline_tables and not md_tables:
        return {"skipped": True, "reason": "no_tables"}
    if not baseline_tables:
        return {"skipped": True, "reason": "baseline_missing"}

    candidates: list[tuple[int, int, float]] = []
    for i, t_md in enumerate(md_tables):
        for j, t_base in enumerate(baseline_tables):
            s = _jaccard(t_md, t_base)
            if s >= JACCARD_THRESHOLD:
                candidates.append((i, j, s))
    candidates.sort(key=lambda x: x[2], reverse=True)

    used_md: set[int] = set()
    used_base: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, j, _ in candidates:
        if i in used_md or j in used_base:
            continue
        pairs.append((i, j))
        used_md.add(i)
        used_base.add(j)

    tp = fp = fn = 0
    for i, j in pairs:
        r = pair_metrics(md_tables[i], baseline_tables[j])
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]

    for i, t in enumerate(md_tables):
        if i not in used_md:
            fp += sum(cell_multiset(t).values())
    for j, t in enumerate(baseline_tables):
        if j not in used_base:
            fn += sum(cell_multiset(t).values())

    result = _prf_from_counts(tp, fp, fn)
    result["tp"] = tp
    result["fp"] = fp
    result["fn"] = fn
    result["tables_matched"] = len(pairs)
    result["tables_total_baseline"] = len(baseline_tables)
    result["skipped"] = False
    return result
