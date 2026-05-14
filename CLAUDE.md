# CLAUDE.md — 项目文档

## 项目概述

本科毕业论文实验代码库，研究题目：

**《基于结构化思维链的零样本对话状态追踪研究：面向系统性幻觉与漏判的诊断与修复》**

网络环境是澳大利亚悉尼，不要使用任何适配中国的环境。

---

## 研究叙事框架

**软件工程视角的核心命题**：LLM 在零样本 DST 任务中存在两类系统性故障——幻觉填值（假阳性）和隐含漏判（假阴性）。本研究将其建模为一个标准的工程诊断-修复流程：

```
① 故障分类  →  标准 prompting 下 GPT-4o-mini 的实际错误分布
② 根因定位  →  两类错误的认知原因：缺乏显式分步推理
③ 靶向修复  →  四步结构化 CoT，每一步对应一类故障
④ 单元验证  →  消融实验：逐步移除各步骤，观察对应故障率变化
⑤ 端到端验证 → 主实验 + 推理忠实性分析
```

**对 CoTE（2403.04656）的定位区分**（必须在论文中明确）：
- CoTE：有监督微调 + 事后生成解释 → 不涉及零样本跨域
- 本研究：零参数 prompting + 事前分步推理 → 专门解决零样本跨域门控错误

---

## 数据集与评测设定

**数据集**：MultiWOZ 2.1（1907.01669），零样本跨域设置

**零样本定义（严格）**：源域 hotel/restaurant/attraction/train 设计提示，目标域 taxi 零标注。few-shot exemplars 来自源域对话，不含任何 taxi 标注。

**主评测集**：`data/processed/test_taxi.json`（1573 turns / 198 dialogs，dialogues 模式）

**评测指标**：JGA（主指标）/ Slot F1 / Precision / Recall / 错误细分（hallucinated_slot / missing_slot / wrong_value）

**模型**：GPT-4o-mini，OpenAI API，无参数更新。

---

## 核心方法：四步结构化 CoT

```
Step 1  领域激活检测  → 当前对话是否涉及目标领域？
                         靶向故障：跨域溢出幻觉（非 taxi 对话产生 taxi 填值）

Step 2  显式信息搜索  → 对话中是否直接出现了该槽位的词语？
                         靶向故障：值提取错误

Step 3  隐含意图推断  → 若无显式提及，是否有间接暗示？
                         靶向故障：隐含漏判（假阴性）

Step 4  None 值决策   → 综合上述证据，无充分依据则强制输出 none
                         靶向故障：幻觉填值（假阳性）
```

步骤文案使用概念引用（EXPLICIT evidence / IMPLICIT evidence），不硬编码步骤编号（2026-04-23 已修复悬空引用问题）。

---

## 实验设计（修订版，8 组条件）

### 主实验条件（实验一 + 横向比较用）

| 变体名 | 描述 | 用途 |
|---|---|---|
| `standard` | 纯 JSON 输出，无推理 | 基线 |
| `fn_style` | 结构化 Schema 描述（含 slot type/desc/values），无推理步骤 | **新增**：测试"schema 结构"本身的贡献 |
| `cot_basic` | 自由 "Let's think step by step" | 弱 CoT 基线（Kojima et al. 2022） |
| `cot_full` | 完整四步结构化 CoT | 主方法 |

**fn_style 的研究价值**：若 fn_style ≈ standard < cot_full → 推理步骤是核心；若 fn_style ≈ cot_basic < cot_full → 结构化信息格式也有贡献。两种结果均可讲出清晰故事。fn_style 设计参照 FnCTOD（2402.10466）的 schema 呈现方式，但不使用函数调用 API。

### 消融实验条件（实验三用）

| 变体名 | 保留步骤 | 靶向故障验证 |
|---|---|---|
| `ab_no_s1` | S2+S3+S4 | Step 1 对跨域溢出幻觉的贡献 |
| `ab_no_s2` | S1+S3+S4 | Step 2 对值提取错误的贡献 |
| `ab_no_s3` | S1+S2+S4 | Step 3 对隐含漏判的贡献 |
| `ab_no_s4` | S1+S2+S3 | Step 4 对幻觉填值的贡献（核心论点） |

**重点观测**：`ab_no_s4` 的 `hallucinated_slot` 密度应显著高于 `cot_full`；`ab_no_s1` 的跨域错误矩阵（`domain_error_breakdown`）应出现 taxi 域假阳性上升。

### 新增分析模块

**实验零（错误诊断，低成本）**：
- 对 `preds_standard_full.json` 运行 `--target_domain taxi` 评测
- 获取 GPT-4o-mini 在 standard 条件下的实际错误分布
- 与 TransferQA（T5 模型）的 37.54%/42.25% 对比，验证动机的跨模型适用性

**推理忠实性分析（50 条手工标注）**：
- 从 cot_full 预测中随机抽 50 个正确预测
- 手工核验推理链引用的"证据"是否真实存在于对话文本
- 四象限分类：忠实推理→正确 / 虚假推理→正确 / 忠实推理→错误 / 虚假推理→错误
- 量化"模型究竟是真推理还是事后合理化"

---

## 论文大纲

```
第一章  绪论
  1.1  研究背景：任务型对话系统与 DST 的工程挑战
  1.2  问题定义：零样本跨域场景下的系统性故障
  1.3  研究贡献
  1.4  论文结构

第二章  相关工作
  2.1  DST 方法演进（TRADE → T5DST → TransferQA）
  2.2  LLM-based DST（ChatGPT / LDST / FnCTOD）
  2.3  思维链提示技术（Few-Shot CoT / Zero-Shot CoT / CoTE）
  2.4  研究空白与本文定位

第三章  研究方法
  3.1  问题建模与实验框架
  3.2  Standard Prompting 故障诊断
  3.3  四步 CoT 的设计原理（步骤↔故障类型的靶向映射）
  3.4  Prompt 工程实现
  3.5  系统架构

第四章  实验设计
  4.1  数据集与零样本设定
  4.2  实验条件（8 组变体）
  4.3  评测指标体系
  4.4  实现细节

第五章  实验结果与分析
  5.1  故障诊断基准：Standard Prompting 错误分布
  5.2  主实验：CoT 有效性验证（4 条件对比含 fn_style）
  5.3  消融实验：步骤贡献度量化
  5.4  推理忠实性分析
  5.5  错误案例研究

第六章  讨论与结论
  6.1  核心发现
  6.2  研究局限（单域评测 / GPT-4o-mini 特异性）
  6.3  未来工作
  6.4  结论
```

---

## 关键约束与注意事项

**模型**：GPT-4o-mini，不进行任何参数更新。

**零样本边界**：任何 taxi 域标注数据不可用于 prompt 设计或示例选取。

**性能预期**：目标是在零样本 LLM 同范式内（vs ChatGPT/LDST/FnCTOD）体现结构化 CoT 的优势，并通过消融和忠实性分析提供机制解释，不需要超越 FnCTOD 整体 SOTA。

**对 CoTE 的态度**：承认 CoTE 证明了 CoT 对 DST 有效（有监督），本文的差异在于零样本设定 + 针对性步骤设计 + 忠实性分析。

---

## 项目结构

```
PROJECT/
├── data/
│   ├── raw/            # MultiWOZ 2.1 原始数据
│   └── processed/      # train/val/test.json + test_taxi.json
├── src/
│   ├── prompt_builder.py   # prompt 变体（需新增 fn_style）
│   ├── inference.py        # GPT-4o-mini 调用，断点续传
│   ├── parser.py           # JSON 抽取 + 非法 slot 过滤
│   ├── evaluator.py        # JGA / Slot F1 / 错误细分
│   ├── ablation_runner.py  # 多 variant 套件
│   ├── compare_ablations.py# MD/CSV/JSON 对比表
│   ├── dst_pipeline.py     # 顶层 CLI
│   ├── filter_taxi_subset.py
│   └── utils.py
├── results/            # 实验输出
├── notebooks/          # 分析笔记本（忠实性标注工具）
├── prompts/            # export_prompts.py 的输出目录（人工核查 prompt 文本）
├── relative_researchs/ # 17 篇参考文献 PDF（CoTE / FnCTOD / TransferQA 等）
├── log.md              # 实验日志（2026-04-23 代码审查 + 修复记录）
├── process.md          # 实验进度计划表（MECE，含命令、成本、状态）
└── CLAUDE.md
```

---

## 当前进度

### 已完成 ✅
- 数据层：MultiWOZ 2.1 处理、test_taxi.json（1573 turns）
- Pipeline：7 种 prompt variant + evaluator v4 + ablation_runner
- 代码修复（2026-04-23）：悬空引用、零样本口径
- 冒烟数据：pilot_50.json / preds_standard_full.json（300条全域 standard）

### 待做（按优先级）

**P0 — 跑前准备（低成本，解锁动机链）**
1. 对 `preds_standard_full.json` 跑 taxi 评测 → 获取 standard 错误分布
2. 在 `prompt_builder.py` 新增 `fn_style` variant

**P1 — 主实验（烧 API）**
3. 跑全量 ablation（8 组 × taxi 子集）：`--suite all --target_domain taxi --tag taxi_full`

**P2 — 深度分析（手工工作）**
4. 50 条 cot_full 正确预测的推理忠实性手工标注

**P3 — 论文写作**
5. 按大纲撰写各章

---

## 关键命令速查

```bash
# P0：分析现有 standard 预测（taxi 口径）
python src/evaluator.py --input results/preds_standard_full.json --target_domain taxi

# P1：全量 ablation（含新增 fn_style，8 组变体）
python src/dst_pipeline.py --mode ablation --suite all --target_domain taxi --tag taxi_full

# 生成对比表
python src/compare_ablations.py --tag taxi_full

# 导出所有 prompt 文本（供人工核查 fn_style 内容）
python src/export_prompts.py
```
