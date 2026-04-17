"""文本指标：字符级 LCS-based P/R/F1 + 编辑距离。

字符级比词级更适合中文；英文另外由 `word_metrics` 补一组词级指标。
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def _lcs_length(a: str, b: str) -> int:
    """LCS 长度 = SequenceMatcher 所有匹配块 size 之和。"""
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def char_metrics(baseline: str, md: str) -> dict[str, Any]:
    """基于字符级 LCS 的 precision / recall / f1。

    两边都为空 → {"skipped": True}
    md 或 baseline 为空而另一边非空 → P=R=F1=0
    """
    if not baseline and not md:
        return {"skipped": True}
    if not md:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "skipped": False}
    if not baseline:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "skipped": False}
    lcs = _lcs_length(baseline, md)
    precision = lcs / len(md)
    recall = lcs / len(baseline)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "skipped": False,
        "_tp": lcs,
        "_fp": len(md) - lcs,
        "_fn": len(baseline) - lcs,
    }


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离，滚动数组 DP。短文本足够快。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


def edit_ratio(baseline: str, md: str) -> float:
    """edit_distance / len(baseline)。baseline 为空时返回 0.0。"""
    if not baseline:
        return 0.0
    return edit_distance(baseline, md) / len(baseline)


def word_metrics(baseline: str, md: str) -> dict[str, Any]:
    """英文按空白切词后跑 LCS。中文不建议用（分词会引入额外噪音）。"""
    b_tokens = baseline.split()
    m_tokens = md.split()
    if not b_tokens and not m_tokens:
        return {"skipped": True}
    if not m_tokens or not b_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "skipped": False}
    matcher = SequenceMatcher(a=b_tokens, b=m_tokens, autojunk=False)
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    precision = lcs / len(m_tokens)
    recall = lcs / len(b_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "skipped": False,
        "_tp": lcs,
        "_fp": len(m_tokens) - lcs,
        "_fn": len(b_tokens) - lcs,
    }
