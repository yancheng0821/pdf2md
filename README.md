# PDF→Markdown 精确转换工具

将 PDF 文件**原封不动**转换为 Markdown，准确率接近 100%。支持文本型和扫描型 PDF。

## 核心特性

- **文本页零成本**：PyMuPDF 本地提取（`find_tables` 识别表格 + `dict` 字号推断标题），不调 API
- **扫描页视觉OCR**：仅扫描/乱码页走视觉模型，最小化 API 调用
- **精确转换**：逐字逐句保留原文，不做任何省略、改写或概括
- **两阶段解耦**：提取（阶段1）和合并（阶段2）独立运行
- **智能清理**：可选 LLM 清理重复页眉页脚、合并跨页断裂表格
- **质量校验**：自动检测数字丢失、内容缺失、乱码残留
- **并发处理**：OCR 并发（默认8路）+ 文件级并发（默认3路）
- **断点续跑**：已完成的文件自动跳过
- **成本预估**：`--dry-run` 预估 token 和费用

## 快速开始

```bash
# 安装依赖
pip install pymupdf openai python-dotenv

# 配置 API Key
cp .env.example .env  # 编辑填入你的 key

# 转换PDF
python parse_pdf.py 报告.pdf

# 处理整个文件夹
python parse_pdf.py ~/reports/ -o ./output
```

## 成本模型

| 页面类型 | 提取方式 | API调用 |
|---------|---------|--------|
| 文本页（≥30字且无乱码） | PyMuPDF 本地提取 | **免费** |
| 扫描/乱码页 | 视觉模型 OCR | 需API |
| 合并清理（可选） | LLM文本处理 | 需API（`--no-cleanup` 可省） |

```bash
# 预估成本
python parse_pdf.py ~/reports/ --dry-run

# 全零API成本（仅限纯文本PDF）
python parse_pdf.py ~/reports/ --no-cleanup
```

## 用法详解

### 基本参数

```bash
python parse_pdf.py <输入路径> [选项]

选项：
  -o, --output          输出目录（默认：输入路径同级的 output/）
  -p, --provider        文本模型，用于合并清理（默认 openai）
  -v, --vision          视觉模型，用于扫描页OCR（默认同 -p）
  --no-cleanup          跳过LLM清理，直接拼接（零API成本）
  --ocr-concurrency     OCR并发数（默认 8）
  --file-concurrency    文件级并发数（默认 3）
  --chunk-chars         合并分块大小（默认 24000）
  --force               强制重新处理
  --stage               运行阶段: all | extract | merge
  --dry-run             预估成本
```

### 阶段控制

```bash
# 只跑提取（阶段1）
python parse_pdf.py ~/reports/ --stage extract

# 只跑合并（阶段2，复用已有的 .pages.md）
python parse_pdf.py ~/reports/ --stage merge
```

### 推荐搭配

```bash
# OCR 用 5.4（最准），清理用 5.4-mini（省钱）
python parse_pdf.py report.pdf -v 5.4 -p openai
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `文件名.md` | 最终 Markdown（与原PDF内容一致） |
| `文件名.pages.md` | 逐页提取结果（阶段1产物，可复用） |
| `文件名.errors.txt` | 失败页面记录（仅在有错误时生成） |

## 处理流程

```
PDF 输入
  │
  ├─ 逐页分类
  │   ├─ 文本页 → PyMuPDF本地提取（find_tables + 字号推断标题）→ Markdown   [免费]
  │   └─ 扫描/乱码页 → 渲染高清图片 → 视觉模型OCR → Markdown              [需API]
  │
  ├─ 阶段1产物：逐页Markdown → .pages.md（持久化，可复用）
  │
  └─ 阶段2：拼接页面 → 可选LLM清理（去重复页眉、合并跨页表格）→ .md
```

## 支持的模型

| 名称 | 模型 | 推荐用途 |
|------|------|---------|
| `openai` | gpt-5.4-mini（默认） | 性价比最优 |
| `5.4` | gpt-5.4 | 最高准确率 |
| `5.4-nano` | gpt-5.4-nano | 最快最便宜 |
| `4o-mini` | gpt-4o-mini | 备用 |
