# 实验进度计划表

> 论文：《基于结构化思维链的零样本对话状态追踪研究：面向系统性幻觉与漏判的诊断与修复》
> 最后更新：2026-05-14

---

## 总览仪表盘

| 阶段 | 名称 | 任务数 | 状态 | API 成本 | 预估工时 |
|---|---|---|---|---|---|
| **Phase 0** | 代码准备 | 3 | 🔲 未开始 | $0 | 1–2 h |
| **Phase 1** | 基线错误诊断 | 2 | 🔲 未开始 | $0 | 0.5 h |
| **Phase 2A** | Pilot 验证 | 3 | 🔲 未开始 | ~$1.5 | 1 h |
| **Phase 2B** | 全量实验 | 2 | 🔲 未开始 | ~$8–10 | 2–3 h |
| **Phase 3** | 深度分析 | 4 | 🔲 未开始 | $0 | 4–5 h |
| **Phase 4** | 论文撰写 | 11 | 🔲 未开始 | $0 | 10–14 天 |

**已完成的前置工作（不在计划内）**：数据预处理 / 7 种 prompt variant / evaluator v4 / ablation_runner / test_taxi.json（1573 turns）/ preds_standard_full.json（300 条 standard 全域预测）

**硬性依赖链**：
```
Phase 0 → Phase 1 → Phase 2A → Phase 2B → Phase 3 → Phase 4（第五章）
                                                      ↑
                          Phase 4（第二、三、四章）可在 Phase 2B 之前并行开始
```

---

## Phase 0：代码准备（无 API，$0）

目标：在跑任何实验之前，让代码库处于完全就绪状态。

---

### T0.1 新增 `fn_style` prompt 变体

| 项 | 内容 |
|---|---|
| **为什么** | 测试"提供更丰富的 schema 描述（无推理步骤）"是否本身就能提升性能；是 fn_style vs cot_full 对比的前提 |
| **输入** | `src/prompt_builder.py` 现有代码 |
| **输出** | `src/prompt_builder.py` 新增 `fn_style` 变体 |
| **工时** | 45–60 min |
| **状态** | 🔲 |

**具体改动要求**：

fn_style 参照 FnCTOD（2402.10466）的 Schema 呈现风格：为每个槽位提供 `type`（string / enum / time）、`optional: true`、`description`、`allowed_values`（若为类别型），但**不要求任何推理步骤**，直接输出 JSON。

在 `prompt_builder.py` 中需要：
1. 新增 `build_slot_list_fn_style()` 函数，以结构化参数格式（`{slot}: ({type}, optional) — {desc} [allowed: ...]`）渲染 schema
2. 在 `VARIANTS` 中新增：
   ```python
   "fn_style": {
       "steps": [],
       "style": "fn",   # 新 style 类型
       "description": "Function-Calling Style (rich schema, no reasoning)",
   }
   ```
3. 在 `build_system_prompt()` 中处理 `style == "fn"` 的分支，调用 `build_slot_list_fn_style()`
4. 在 `SUITES` 中（`ablation_runner.py`）将 `fn_style` 加入 `"main"` 套件

---

### T0.2 Prompt 人工核查

| 项 | 内容 |
|---|---|
| **为什么** | 确认 fn_style 的 schema 格式正确；确认 8 个变体均可成功构建且互不混淆 |
| **输入** | T0.1 完成后的 prompt_builder.py |
| **输出** | `prompts/` 目录下 8 份导出文本（`export_prompts.py` 产物） |
| **命令** | `python src/export_prompts.py` |
| **工时** | 20–30 min（运行 + 肉眼逐份检查） |
| **前置** | T0.1 ✅ |
| **状态** | 🔲 |

**核查要点**：
- fn_style 的 schema 含 type / optional / description，无 `<reasoning>` 块
- cot_full 的步骤从 1 开始连续编号，Step 3/4 文案无"Step 2"等硬编码引用
- ab_no_s\* 系列的步骤编号与保留步骤数一致（例如 ab_no_s1 只有 Step 1/2/3）

---

### T0.3 Smoke Test fn_style（5 条样本）

| 项 | 内容 |
|---|---|
| **为什么** | 以最低成本验证 fn_style 能正常调用 GPT-4o-mini 并返回可解析的 JSON |
| **输入** | test_taxi.json 前 5 条 |
| **命令** | `python src/dst_pipeline.py --mode single --variant fn_style --max_samples 5 --tag smoke` |
| **输出** | `results/preds_fn_style_smoke.json`（5 条预测，确认无解析报错） |
| **API 成本** | ~5 次调用，< $0.01 |
| **前置** | T0.2 ✅ |
| **状态** | 🔲 |

---

## Phase 1：基线错误诊断（无 API，$0）

目标：在运行新实验之前，先从已有 standard 预测中提取 taxi 域的错误分布，作为全文的动机基准（"论文第一张数据表"）。

---

### T1.1 Standard 基线的 taxi 域错误分析

| 项 | 内容 |
|---|---|
| **为什么** | 验证 GPT-4o-mini 在 standard 条件下确实呈现两类系统性门控错误，为 CoT 设计提供第一手动机数据；避免仅借用 T5 模型（TransferQA）的错误统计 |
| **输入** | `results/preds_standard_full.json`（300 条全域 standard 预测，已存在） |
| **命令** | `python src/evaluator.py --input results/preds_standard_full.json --target_domain taxi` |
| **输出** | 控制台打印 + `results/eval_report_standard_taxi.json`（如 evaluator 支持 `--output` 参数，否则手动保存输出） |
| **工时** | 15 min（运行 + 记录） |
| **状态** | 🔲 |

**期望从输出中记录**（填入论文表格）：

| 指标 | standard 全域（已知） | standard taxi 域（待测） |
|---|---|---|
| JGA | 28.0% | ？ |
| Slot F1 | 79.77% | ？ |
| hallucinated_slot 密度 | — | ？ |
| missing_slot 密度 | — | ？ |
| wrong_value 密度 | — | ？ |

---

### T1.2 与 TransferQA 错误分布对比

| 项 | 内容 |
|---|---|
| **为什么** | 确认 GPT-4o-mini（LLM）与 T5（TransferQA）呈现相似的错误模式，从而让 CoT 设计动机跨模型成立 |
| **输入** | T1.1 输出 + TransferQA 原论文数字（37.54% 幻觉 / 42.25% 漏判 / 20.21% 值错） |
| **输出** | 论文第五章 5.1 节的"动机验证表"（手动整理，可存为 `results/motivation_table.md`） |
| **工时** | 20 min |
| **前置** | T1.1 ✅ |
| **状态** | 🔲 |

**如果 GPT-4o-mini 的错误分布与 TransferQA 明显不同**：在论文中如实呈现，并说明"LLM 在幻觉方向的倾向更强/更弱，但两类门控错误仍是主要失效模式"——这本身就是一个有价值的发现。

---

## Phase 2A：Pilot 验证（API 消耗 ~$1.5）

目标：以小规模样本（200 条）验证 8 组变体的对比表结构和错误分类方向，确认全量实验值得跑。

---

### T2A.1 跑 Pilot（200 × 8 = 1600 次调用）

| 项 | 内容 |
|---|---|
| **命令** | `python src/dst_pipeline.py --mode ablation --suite all --target_domain taxi --max_samples 200 --tag taxi_pilot200` |
| **输出** | `results/preds_*_taxi_pilot200.json`（8 份）+ `results/ablation_summary_taxi_pilot200.md` |
| **API 成本** | ~1600 次 → ~$1.5 |
| **墙钟时间** | ~30–45 min |
| **前置** | T0.3 ✅ |
| **状态** | 🔲 |

---

### T2A.2 Pilot 结果核查

| 项 | 内容 |
|---|---|
| **为什么** | 确认结果方向符合预期，再决定是否进行全量实验 |
| **输入** | `results/ablation_summary_taxi_pilot200.md` |
| **工时** | 20–30 min |
| **前置** | T2A.1 ✅ |
| **状态** | 🔲 |

**6 条 Go/No-Go 检查**（全部通过才进入 2B）：

- [ ] `cot_full` JGA > `standard` JGA（主方法有效性）
- [ ] `cot_full` JGA > `fn_style` JGA（推理步骤的贡献 > schema 格式的贡献）
- [ ] `ab_no_s4` 的 `hallucinated_slot` 密度 > `cot_full`（Step 4 抑幻觉验证）
- [ ] `ab_no_s1` 的跨域 hallucination 密度 > `cot_full`（Step 1 过滤跨域溢出验证）
- [ ] `cot_basic` 介于 `standard` 和 `cot_full` 之间（弱 CoT 基线符合预期）
- [ ] `fn_style` JGA 高于 `standard`（schema 描述有独立价值）或等同（schema 无独立价值）——两种结果均可接受，只要方向清晰

**如果某条检查不符合预期**：在 `log.md` 记录发现，分析原因（可能需要微调 fn_style 的 prompt，或检查 evaluator 对 taxi 槽的过滤逻辑），再重跑 pilot。

---

### T2A.3 记录 Pilot 结论（更新 log.md）

| 项 | 内容 |
|---|---|
| **输出** | `log.md` 新增一节（日期 + 6 条核查结果 + Go/No-Go 决定） |
| **工时** | 15 min |
| **前置** | T2A.2 ✅ |
| **状态** | 🔲 |

---

## Phase 2B：全量实验（API 消耗 ~$8–10）

**前置**：Phase 2A 全部通过

---

### T2B.1 跑全量（1573 × 8 = 12,584 次调用）

| 项 | 内容 |
|---|---|
| **命令** | `python src/dst_pipeline.py --mode ablation --suite all --target_domain taxi --tag taxi_full` |
| **输出** | `results/preds_*_taxi_full.json`（8 份）+ `results/ablation_summary_taxi_full.md` + `.csv` + `.json` |
| **API 成本** | ~12,584 次 → ~$8–10（GPT-4o-mini 估算） |
| **墙钟时间** | 1.5–2.5 h（支持断点续传，可分批） |
| **状态** | 🔲 |

> 断点续传：若中途中断，重新执行同命令即可，inference.py 会自动跳过已完成的 `(dial_id, turn_id)` 对。

---

### T2B.2 生成最终对比表

| 项 | 内容 |
|---|---|
| **命令** | `python src/compare_ablations.py --tag taxi_full` |
| **输出** | `results/comparison_taxi_full.md`（论文表格素材）+ `.csv`（matplotlib 用） |
| **前置** | T2B.1 ✅ |
| **工时** | 5 min |
| **状态** | 🔲 |

---

## Phase 3：深度分析（无 API，手工为主）

目标：从实验数据中提取论文第五章 5.4、5.5 节所需的质性分析素材。

**前置**：Phase 2B ✅

---

### T3.1 推理忠实性标注准备

| 项 | 内容 |
|---|---|
| **为什么** | 验证 CoT 推理链的"证据引用"是否真实存在于对话文本，回答"模型是真推理还是事后合理化" |
| **输入** | `results/preds_cot_full_taxi_full.json` |
| **输出** | `notebooks/faithfulness_annotation.ipynb`（含样本加载 + 标注界面 + 统计分析） |
| **工时** | 45–60 min（搭建 notebook） |
| **状态** | 🔲 |

**Notebook 功能要求**：
1. 随机抽取 50 条 `jga == true`（正确预测）样本
2. 每条展示：对话历史 / CoT 推理链 / 预测 belief / 金标 belief
3. 标注字段：`reasoning_faithful`（True/False），`note`（可选说明）
4. 自动保存到 `results/faithfulness_annotation.csv`

---

### T3.2 50 条推理链人工标注

| 项 | 内容 |
|---|---|
| **标注规则** | 推理链中每次引用"某turn提到了X"，检查该 turn 是否真实包含 X。全部引用均真实 → Faithful；存在任何虚构引用 → Unfaithful |
| **输入** | T3.1 的 notebook |
| **输出** | `results/faithfulness_annotation.csv`（50 行，含标签） |
| **工时** | 2–3 h |
| **前置** | T3.1 ✅ |
| **状态** | 🔲 |

**四象限分类（标注完成后自动统计）**：

| | 预测正确（jga=True） | 预测错误（jga=False） |
|---|---|---|
| **推理忠实** | ① 理想情形 | ③ 推理对但结论错 |
| **推理虚假** | ② 侥幸正确 | ④ 双重错误 |

- ① 占比高 → CoT 通过真实推理达到正确答案（方法可解释性强）
- ② 占比高 → 模型在"事后合理化"，CoT 结果可信度存疑

---

### T3.3 步骤贡献↔错误类型映射表整理

| 项 | 内容 |
|---|---|
| **为什么** | 将消融实验数字与设计意图对应，构成论文核心论证 |
| **输入** | `results/ablation_summary_taxi_full.md` 中各变体的 `error_breakdown` |
| **输出** | `results/step_error_mapping.md`（手动整理，论文表格素材） |
| **工时** | 30–45 min |
| **前置** | T2B.2 ✅ |
| **状态** | 🔲 |

**目标表格结构**：

| 消融变体 | 相对 cot_full Δhallucinated | Δmissing | Δwrong_value | 设计印证 |
|---|---|---|---|---|
| ab_no_s1 | ↑ 跨域幻觉 | — | — | Step 1 过滤跨域溢出 ✓ |
| ab_no_s2 | — | — | ↑ | Step 2 减少值提取错误 ✓ |
| ab_no_s3 | — | ↑ | — | Step 3 补隐含漏判 ✓ |
| ab_no_s4 | ↑ 整体幻觉 | — | — | Step 4 None 验证 ✓ |

---

### T3.4 案例研究样本选取

| 项 | 内容 |
|---|---|
| **为什么** | 论文 5.5 节的案例研究，让定量结论有直观的定性支撑 |
| **输入** | 全量预测 JSON + faithfulness 标注结果 |
| **输出** | 4 类典型样本各 1–2 条，存入 `results/case_studies.md` |
| **工时** | 45–60 min |
| **前置** | T3.2 ✅ T3.3 ✅ |
| **状态** | 🔲 |

**4 类目标案例**：

| 类型 | 选取标准 | 用途 |
|---|---|---|
| A. CoT 抑制幻觉 | standard 填值 / cot_full 正确输出 none | 展示 Step 4 效果 |
| B. CoT 修复漏判 | standard 漏 / cot_full 正确推断隐含 | 展示 Step 3 效果 |
| C. 虚假推理侥幸正确 | faithfulness=False 且 jga=True | 展示方法局限性 |
| D. 跨域溢出 | ab_no_s1 幻觉 / cot_full 正确 | 展示 Step 1 效果 |

---

## Phase 4：论文撰写

按**写作顺序**排列（非章节顺序）。第五章依赖实验数据，其余章节可提前并行。

**可提前开始（依赖 Phase 2B 前）**：第二章、第三章、第四章

---

### T4.1 第二章：相关工作

| 项 | 内容 |
|---|---|
| **字数目标** | 3000–4000 字 |
| **四节结构** | DST 方法演进 → LLM-based DST → CoT 技术 → 研究空白 |
| **写作重点** | 明确本文与 CoTE（2403.04656）的差异；将 FnCTOD 定位为"同范式竞争者"；表格展示各方法的范式对比 |
| **前置** | 无（可立即开始） |
| **预估工时** | 2–3 天 |
| **状态** | 🔲 |

---

### T4.2 第三章：研究方法

| 项 | 内容 |
|---|---|
| **字数目标** | 2500–3500 字 |
| **四节结构** | 问题建模 → 故障诊断动机 → 四步 CoT 设计（含步骤↔故障类型映射图） → Prompt 工程细节 |
| **关键图表** | 图：四步推理链流程图（步骤→故障类型箭头）；表：8 种 prompt 变体对比 |
| **前置** | T0.1 ✅（fn_style 定稿后） |
| **预估工时** | 2–3 天 |
| **状态** | 🔲 |

---

### T4.3 第四章：实验设计

| 项 | 内容 |
|---|---|
| **字数目标** | 1500–2000 字 |
| **内容** | 数据集说明（MultiWOZ 2.1）→ 零样本设定（taxi 子集 1573 turns）→ 8 种实验条件定义 → 评测指标（JGA / Slot F1 / 错误细分） → 实现细节（温度=0，断点续传，费用估算） |
| **前置** | Phase 2B 完成（以便填写实际 API 调用量和费用） |
| **预估工时** | 1 天 |
| **状态** | 🔲 |

---

### T4.4 第五章 §5.1：Standard 错误诊断

| 项 | 内容 |
|---|---|
| **字数目标** | 600–800 字 |
| **核心表格** | GPT-4o-mini standard 错误分布 vs TransferQA（T5）错误分布对比 |
| **结论** | 证明 LLM 在零样本 DST 场景下复现了两类门控错误，为 CoT 设计提供第一手动机 |
| **前置** | T1.2 ✅ |
| **预估工时** | 0.5 天 |
| **状态** | 🔲 |

---

### T4.5 第五章 §5.2：主实验结果

| 项 | 内容 |
|---|---|
| **字数目标** | 800–1000 字 |
| **核心表格** | 4 条件（standard / fn_style / cot_basic / cot_full）× JGA / Slot F1 / 错误三分 |
| **关键论点** | ① cot_full > cot_basic > standard（CoT 有效）；② cot_full > fn_style（推理步骤 > schema 格式）；③ 引用 FnCTOD/ChatGPT 文献数字对齐同范式位置 |
| **前置** | T2B.2 ✅ |
| **预估工时** | 1 天 |
| **状态** | 🔲 |

---

### T4.6 第五章 §5.3：消融实验

| 项 | 内容 |
|---|---|
| **字数目标** | 1000–1200 字 |
| **核心表格** | T3.3 的步骤贡献↔错误类型映射表；柱状图：各变体 hallucinated/missing/wrong_value 密度 |
| **关键论点** | 每步消融只影响其靶向错误类型，其他错误类型基本不变 → 设计的精准性 |
| **前置** | T3.3 ✅ |
| **预估工时** | 1–1.5 天 |
| **状态** | 🔲 |

---

### T4.7 第五章 §5.4：推理忠实性分析

| 项 | 内容 |
|---|---|
| **字数目标** | 600–800 字 |
| **核心表格** | 四象限频率统计 + 典型忠实/虚假推理对比示例 |
| **关键论点** | 忠实推理占正确预测的多数 → 方法的可解释性有实证基础；同时坦诚虚假推理侥幸正确的比例，作为 limitation 铺垫 |
| **前置** | T3.2 ✅ |
| **预估工时** | 1 天 |
| **状态** | 🔲 |

---

### T4.8 第五章 §5.5：错误案例研究

| 项 | 内容 |
|---|---|
| **字数目标** | 500–700 字 |
| **内容** | T3.4 中 4 类典型案例的展示与分析，每类 1 个对话片段 + reasoning 摘录 + 分析 |
| **前置** | T3.4 ✅ |
| **预估工时** | 0.5 天 |
| **状态** | 🔲 |

---

### T4.9 第六章：讨论与结论

| 项 | 内容 |
|---|---|
| **字数目标** | 1500–2000 字 |
| **四节** | 核心发现总结 → 研究局限（单域 / GPT-4o-mini 特异性 / categorical 模板未实现）→ 未来工作 → 结论 |
| **前置** | Phase 3 全部 ✅ |
| **预估工时** | 1 天 |
| **状态** | 🔲 |

---

### T4.10 第一章：绪论（最后写）

| 项 | 内容 |
|---|---|
| **字数目标** | 2000–2500 字 |
| **四节** | 研究背景 → 问题定义（工程诊断框架） → 贡献声明（3 条） → 论文结构 |
| **为什么最后写** | 贡献声明要与实际实验结果一致，最后写保证不夸大 |
| **前置** | T4.5 T4.6 T4.7 ✅（知道实验结论后再写） |
| **预估工时** | 1 天 |
| **状态** | 🔲 |

---

### T4.11 格式校对 & 收尾

| 项 | 内容 |
|---|---|
| **内容** | 图表编号统一 / 参考文献格式（ACL 格式）/ 中英文术语一致性 / 摘要（中英双语） / 字数统计 |
| **前置** | T4.1 → T4.10 全部完成草稿 |
| **预估工时** | 1.5–2 天 |
| **状态** | 🔲 |

---

## 附录 A：API 预算明细

| 任务 | 样本量 | 变体数 | 预估调用 | 预估成本 |
|---|---|---|---|---|
| T0.3 Smoke | 5 | 1 | 5 | < $0.01 |
| T2A.1 Pilot | 200 | 8 | 1,600 | ~$1.5 |
| T2B.1 Full | 1,573 | 8 | 12,584 | ~$8–10 |
| **合计** | | | **~14,200** | **~$10–12** |

> 估算基础：GPT-4o-mini，CoT 变体 input ≈ 2,500 tokens，output ≈ 450 tokens；standard ≈ input 1,200 / output 200 tokens。
>
> fn_style 与 standard 接近（无推理输出），不拉高平均成本。

---

## 附录 C：执行过程中发现的问题与待办（2026-05-14 新增）

| # | 问题 | 严重度 | 状态 | 处理 |
|---|---|---|---|---|
| P1 | Pilot (tag=`taxi_pilot200`) 与 Full (tag=`taxi_full`) 文件名不同，断点续传不复用 → 前 200 条会重跑（白烧 ~$0.5） | 中 | ✅ 已解决 | Pilot 完成后将 `preds_*_taxi_pilot200.json` 改名为 `preds_*_taxi_full.json`（含 eval/log），Full 启动后从第 201 条接着跑 |
| P2 | `JGA` 在 taxi 投影下含空对空匹配虚高（standard 全域 92% vs taxi 子集 ≈69%） | 中 | 📝 待写作时声明 | 论文 §4.3 / §5.1 明确主指标为 Slot F1，JGA 仅作辅助，并注明零样本投影偏差 |
| P3 | 零样本边界口径：schema 已知 / exemplar 未知，与 FnCTOD/TransferQA 同口径 | 低 | 📝 待写作时声明 | 论文 §3.1 / §4.1 显式声明"无标注 / 无 exemplar 零样本，schema-known" |
| P4 | T2A.2 Go/No-Go 第 5 条"cot_basic 居中"过严 | 低 | ✅ 已处理 | 改为观察项；硬指标只保留 cot_full > standard、cot_full > fn_style、ab_no_s4 ↑hallucination |
| P5 | 断点续传以文件名为 key，对 prompt 内容修改无防御 | 低 | 🔲 未来工程改进 | 暂不修；若未来重跑实验可在 results JSON 内加 `prompt_fingerprint` 字段 |
| P6 | `taxi-arrive by` 槽名含空格，LLM 可能输出 `arriveby` 等变体被 parser 丢弃 | 低 | 🔲 观察中 | Pilot 完成后检查 `wrong_value_by_slot` 是否高频集中在 arrive by/leave at，再决定是否加 alias |
| P7 | T3.2 50 条人工标注耗时（2-3h） | 中 | ✅ 已加自动化 | 新增 `src/auto_faithfulness_check.py` 严谨规则自动产出初版标签，人类只 review，不从头标 |

---

## 附录 B：产出物清单（最终状态）

| 类别 | 文件路径 | 来源任务 |
|---|---|---|
| **实验数据** | `results/preds_*_taxi_full.json`（8 份） | T2B.1 |
| **对比表** | `results/comparison_taxi_full.md / .csv` | T2B.2 |
| **错误诊断** | `results/eval_report_standard_taxi.json` | T1.1 |
| **动机表** | `results/motivation_table.md` | T1.2 |
| **步骤映射** | `results/step_error_mapping.md` | T3.3 |
| **忠实性标注** | `results/faithfulness_annotation.csv` | T3.2 |
| **案例研究** | `results/case_studies.md` | T3.4 |
| **Prompt 导出** | `prompts/`（8 份 .txt） | T0.2 |
| **分析 Notebook** | `notebooks/faithfulness_annotation.ipynb` | T3.1 |
| **论文** | 独立 Word/LaTeX 文件 | Phase 4 |
