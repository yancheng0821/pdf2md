# pdf2md 转换质量评估工具设计

> Date: 2026-04-17
> Status: Approved for planning

## 1. 背景与目标

`pdf2md.py` 已能把 PDF 转成 Markdown（`canonical.md` + `pages/*.md`），目前无任何手段衡量转换质量。本设计给出一套**无需人工标注的自动评估工具**，对 5 份已转换的代表性 PDF 做"**准确率 / 召回率**"对比，输出可驱动后续优化的报告。

**关键认知前提**：没有人工 ground truth，算出的数字是 **"MD 相对程序化基线的一致性"**，不是"MD 相对真相的准确率"。所有报告必须显式披露这一点，避免被误读。

## 2. 范围

**评估对象**（已有转换产物，位于 `~/Desktop/归档/output/`）：

| PDF | 页数 |
|---|---|
| 天津真格天弘 | 45 |
| 昆山峰瑞 | 11 |
| 广州越秀 | 32 |
| IDG China Capital Fund III（英文） | 38 |
| 苏州济峰基金 | 21 |

**非目标**：
- 不重跑 pdf2md（只评估已有产物）。
- 不做大模型辅助评估、不调任何外部 API。
- 不对字段级业务数据做准确率评估（缺标注，本轮不做）。
- 不修改 `pdf2md.py`。

## 3. 评估维度

两条独立流水线，互不依赖：

- **A. 纯文本一致性**：MD 去 Markdown 标记后的纯文本 vs PyMuPDF 抽取的原生文本，字符级 P/R/F1 + 编辑距离。
- **C. 表格单元格一致性**：MD 中的表格解析成 2D 数组 vs pdfplumber 抽取的表格，单元格级 P/R/F1。

## 4. 目录结构

```
pdf-parser/
├── pdf2md.py                     # 不修改
├── eval/                         # 新增
│   ├── __init__.py
│   ├── run_eval.py               # 入口
│   ├── config.yaml               # 5 份 PDF 的路径配置（绝对路径，.gitignore）
│   ├── baseline.py               # PyMuPDF 文本 + pdfplumber 表格
│   ├── md_parser.py              # 解析 pages/*.md 成 (text, tables)
│   ├── metrics_text.py           # 文本 P/R/F1 + 编辑距离
│   ├── metrics_table.py          # 表格单元格 P/R/F1
│   ├── report.py                 # 汇总 + 分页明细 + bad case
│   └── tests/
│       ├── fixtures/             # 手写 baseline/md 对照样本
│       ├── test_text_metrics.py
│       ├── test_table_metrics.py
│       ├── test_md_parser.py
│       ├── test_baseline.py
│       └── test_run_eval_smoke.py
└── eval_output/                  # 新增（.gitignore）
    ├── summary.csv
    ├── summary.md
    └── <pdf-name>/
        ├── per_page.csv
        ├── bad_cases.md
        └── raw.json
```

**约束**：
- 评估器完全旁路 `pdf2md.py`，只读源 PDF 和已有 `canonical.md` / `pages/*.md`。
- 新依赖仅 3 个：`PyMuPDF`、`pdfplumber`、`PyYAML`。
- `config.yaml` 含本机绝对路径，不入库。

## 5. 数据流

每份 PDF 独立处理：

```
源 PDF ─┬─ PyMuPDF.get_text ──→ baseline_text[page_i]
        └─ pdfplumber.extract_tables ──→ baseline_tables[page_i]

pages/00X.md ─┬─ 去 Markdown ──→ md_text[page_i]
              └─ 解析表格 ──→ md_tables[page_i]

        ↓ 归一化（两边共用同一函数）
        ↓ metrics_text + metrics_table（逐页）
        ↓ micro-average 到 per-PDF / 总体
        ↓ report 生成 summary / per_page / bad_cases / raw.json
```

## 6. 基线抽取

**6.1 文本基线（PyMuPDF）**

- `page.get_text("text")`
- 归一化流水线（MD 和基线共用）：
  1. NFKC
  2. 去零宽字符（`\u200b`、`\ufeff` 等）
  3. 压缩连续空白为单个空格
  4. 英文小写化
- 扫描件识别：归一化后文本长度 < 20 字符 → `scanned=True`。

**6.2 表格基线（pdfplumber）**

- `page.extract_tables()`，默认策略。
- 每个表 → `list[list[str]]`，单元格去首尾空白，`None` 转空字符串。
- 合并单元格不做特殊处理（两边都会拆成重复值）。
- `pdfplumber` 对某页抛异常 → 该页 `table_baseline_missing=True`，不中断。

**6.3 MD 侧解析**

- **逐页**读 `pages/001.md` ~ `pages/NNN.md`（非 `canonical.md`），与基线按 page 对齐。
- 文本：用简单正则剥离 Markdown（`#`、`*`、`>`、`-`、链接、图片、表格语法），剩下纯文本过同一归一化。
- 表格：正则匹配 markdown 表格块（`| ... |` + `|---|`），解析成 2D 数组；同时支持 HTML `<table>`（保险起见，目前样本未见）。
- 页数不匹配：缺失的页 `status=page_missing`。

## 7. 指标定义

**7.1 文本（per page，扫描页跳过）**

基于 unicode **字符序列**（不分词）：

- `L = difflib.SequenceMatcher(baseline, md).ratio()` 对应的 LCS 长度
- `Precision = L / len(md_text)`
- `Recall = L / len(baseline_text)`
- `F1 = 2PR/(P+R)`
- `edit_ratio = Levenshtein / len(baseline)`

英文 PDF（IDG）**额外**按空白切词再算一组 `text_P_word / text_R_word / text_F1_word`（字符级在字母表小的语言上会虚高）。

**7.2 表格（per page，基线无表跳过）**

页内多表配对（贪心 Jaccard）：

1. 对每对 (T_md, T_base)，`S = Jaccard(unique_cells_md, unique_cells_base)`。
2. 按 S 降序贪心配对，阈值 `S >= 0.1`。
3. 未配对的 MD 表 → 所有单元格算 FP；未配对的基线表 → 所有单元格算 FN。

配对后单元格级（multiset）：

- `TP = |multiset_md ∩ multiset_base|`
- `FP = |multiset_md| - TP`
- `FN = |multiset_base| - TP`
- P / R / F1 同上

**不按行列位置匹配**，容忍小幅行错位，代价是放过"行错位但内容都在"的情况——本轮接受。

**7.3 聚合**

- **Per-PDF**：把所有页的 TP/FP/FN 加总再算 P/R/F1（**micro-average**，不用 macro）。
- **总体**：5 份 PDF 的 TP/FP/FN 再 micro 一次。

## 8. 报告 Schema

**8.1 `eval_output/summary.csv` + `summary.md`**

列：`pdf, lang, pages_total, pages_scanned, pages_table_missing, text_P, text_R, text_F1, text_edit_ratio, text_P_word, text_R_word, text_F1_word, table_P, table_R, table_F1, tables_total, tables_matched`。中文 PDF 的 `text_*_word` 三列留空（CSV 为空字符串，JSON 为 `null`）。

`summary.md` 同数据 + 一段 TL;DR（自动生成："N 份共 X 页，Y 页扫描件无法评估；文本 F1=…；表格 F1=…；最弱：<pdf> 表格 recall=…"）。

**8.2 `eval_output/<pdf-name>/per_page.csv`**

每行一页：`page, status, baseline_text_len, md_text_len, text_P, text_R, text_F1, text_edit_ratio, baseline_tables, md_tables, table_P, table_R, table_F1, md_method`。

`md_method` 从对应 `manifest.json` 的 `pages[i].method` 取（`native` / `local_ocr` / ...），用于关联"哪种提取方法效果好"。

**8.3 `eval_output/<pdf-name>/bad_cases.md`**

每份 PDF 选 **TOP-3 最差页**。排序键 = 该页存在的 F1 的最小值（文本无分则看表格，表格无分则看文本，两者皆无则该页不参与 bad case 选择）。仅从 `status=ok` 的页里选，升序取前 3。每节包含：

- 页号、指标、`md_method`
- 文本 diff：`difflib.unified_diff` 前 40 行，超长截断
- 表格 diff：表格化列出每个单元格在基线/MD 中的存在情况与 `match / missing_in_md / extra_in_md`

**8.4 `eval_output/<pdf-name>/raw.json`**

所有页所有指标的原始数字，便于后续用其他脚本重分析。schema 与 per_page.csv 一致 + 顶层 per-PDF 聚合指标。

**8.5 控制台输出**

每份 PDF 进度条，结束打印 summary 表；有异常的 PDF 在控制台醒目标注。

## 9. 错误处理

**原则**：一份 PDF / 一页 / 一个表的失败不能拖垮整体评估。

| 情形 | 策略 |
|---|---|
| config 中 PDF 路径不存在 | 加载时 fail-fast，指出缺失行 |
| `pages/*.md` 数量 ≠ PDF 页数 | 缺失页 `status=page_missing`，summary 顶端告警 |
| PyMuPDF 抽某页抛异常 | 该页降级 `scanned`，异常写 `eval_output/<pdf>/errors.log` |
| pdfplumber 抽表抛异常 | 该页 `table_baseline_missing`，不中断 |
| MD 文件编码异常 | `utf-8` 优先，失败 fallback `gbk`，再失败才抛 |
| 某份 PDF 全跑挂 | 其他 4 份继续；summary 中该份用 `FAILED` 占位 |

## 10. 测试策略

`eval/tests/`，pytest。

| 文件 | 覆盖内容 |
|---|---|
| `test_text_metrics.py` | 相同/子序列/超集/完全不同/空基线 fixtures |
| `test_table_metrics.py` | 相同/少行/多列/多表配对场景 |
| `test_md_parser.py` | 从真实 `pages/` 抠的 5~10 页 fixture，验证文本剥离和表格解析 |
| `test_baseline.py` | 冒烟：用最短 PDF（昆山峰瑞 11 页）跑通基线函数，不强断数值 |
| `test_run_eval_smoke.py` | 端到端冒烟：跑 1 份 PDF，验证 4 种产物都非空 |

**不测**：第三方库本身；不对具体 PDF 做"应该 F1=0.91"之类的强断言（被测对象会变）。

## 11. 开发顺序

1. `baseline.py` + `test_baseline.py`
2. `md_parser.py` + `test_md_parser.py`
3. `metrics_text.py` + `metrics_table.py` + tests
4. `report.py`
5. `run_eval.py` + `config.yaml` + 控制台进度
6. 端到端跑 5 份，人眼抽查 `bad_cases.md` 校验脚本合理性

## 12. 成功标准

- 5 份 PDF 全部跑完无崩溃，每份都产出 4 类文件。
- `summary.md` 的 TL;DR 能让人 30 秒内判断"整体好坏 + 最弱的那份"。
- `bad_cases.md` 对每份 PDF 的 TOP-3 差页都展示了可定位的 diff，能指向"应该去改 pdf2md 的哪一块"。
- 评估器本身的 pytest 全绿。
