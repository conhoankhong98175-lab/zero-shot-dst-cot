# 实验日志 —— 消融实验（Ablation Study）预检

> 更新时间：2026-04-23
> 阶段：模块三（Ablation Study）启动前代码审查
> 审查范围：`src/prompt_builder.py`, `src/inference.py`, `src/parser.py`, `src/evaluator.py`, `src/ablation_runner.py`, `src/compare_ablations.py`, `src/dst_pipeline.py`, `src/utils.py`, `data/download_multiwoz.py`

---

## 一、当前代码状态总览

| 模块 | 文件 | 状态 | 备注 |
|---|---|---|---|
| Prompt 构造 | `prompt_builder.py` | 🟡 基本可用，有轻度缺陷 | 7 个 variant 全定义，步骤文本存在跨步引用 |
| 推理主循环 | `inference.py` | ✅ 质量良好 | temperature=0、指数退避、断点续传、原子写 |
| 输出解析 | `parser.py` | ✅ 质量良好 | JSON 代码块 + fallback、非法 slot 过滤 |
| 评测 | `evaluator.py v4` | ✅ 质量良好 | 四层归一化、三类错误细分、域×错误矩阵 |
| 套件编排 | `ablation_runner.py` | ✅ 可用 | 单 variant 失败不影响全局，自动生成对比表 |
| 横向对比 | `compare_ablations.py` | ✅ 可用 | MD / CSV / JSON 三种产物，含 Δ baseline |
| 顶层入口 | `dst_pipeline.py` | ✅ 可用 | pilot / full / single / ablation 四模式 |
| 数据预处理 | `data/download_multiwoz.py` | 🟡 可用但未按零样本裁剪 | 见下文 "关键问题 1" |

结论：**技术上可以启动全量实验**；但在动用 API 预算之前，强烈建议先确认下方 4 条关键问题，否则实验结果可能无法支撑 CLAUDE.md 中宣称的"零样本跨域（taxi 目标域）"这一论文卖点。

---

## 二、已完成的内容（无需返工）

### 实验一（主对比）基础设施
- [x] `standard`：纯 JSON 输出基线（无 reasoning）
- [x] `cot_basic`：自由 "think step by step"
- [x] `cot_full`：结构化四步 CoT（S1+S2+S3+S4）
- [x] 三种 variant 的 prompt 已区分 `style=none / free / structured`
- [x] `_MAX_TOKENS` 差异化：standard=512，CoT=2048（避免推理链被截断）

### 实验二（逐步消融）基础设施
- [x] `ab_no_s1 / ab_no_s2 / ab_no_s3 / ab_no_s4` 四个变体已定义
- [x] 每个变体的 `steps` 列表与 `description` 对齐
- [x] `ablation_runner` 将 `cot_full` 作为 ablation 套件的 baseline 自动纳入

### 评测与错误分析基础设施
- [x] JGA / Slot F1 / Precision / Recall
- [x] 错误类型三分法：`hallucinated_slot` / `missing_slot` / `wrong_value`
  - 对应 TransferQA 论文的假阳性 / 假阴性 / 值错
- [x] 按域 × 错误类型的矩阵（`domain_error_breakdown`），**专门支撑 S1 消融的论文叙事**
- [x] 跨规模可比的 `*_per_100`（每 100 轮的错误密度）
- [x] Δ-vs-baseline（相对 cot_full 的性能差值）

### 工程防御
- [x] 断点续传（按 `dial_id + turn_id` 去重）
- [x] 原子写（`.tmp → rename`）
- [x] 指数退避重试（`RateLimitError` 60s / 其他错误 1/2/4s）
- [x] 每 20 条样本落盘一次
- [x] 单变体失败不阻断其他变体
- [x] `experiment_log.jsonl` 追加式审计日志

### 冒烟测试
- [x] `pilot_50.json`（50 条 cot_full 样本）已跑通
- [x] `preds_standard_full.json`（300 条 standard 样本）已存在，可作为参考基线
- [x] `eval_report.json` 冒烟报告：JGA ≈ 24% / F1 ≈ 83%

---

## 三、关键问题（上线前必须决策）

### 🔴 问题 1：零样本跨域设定未在数据层落实（最关键）

**现状**：
- `data/processed/test.json` 含 **7372 条样本 / 1000 个对话**，覆盖全部 5 个领域
- 其中 **仅 642 条样本涉及 taxi 域**，**6730 条样本完全没有 taxi**
- `evaluator.py` 对所有 5 个领域一视同仁地计算 JGA / Slot F1
- `prompt_builder.SLOT_SCHEMA` 同时暴露全部 5 个领域的 slot 说明

**与 CLAUDE.md 的冲突**：
CLAUDE.md 明确定义："零样本（Zero-Shot Cross-Domain）场景定义：模型在若干源领域（hotel / restaurant / attraction / train）上设计提示，直接在目标领域（taxi）上测试，**目标领域零条标注数据**"。

**影响**：
- 如果按当前代码跑，实验测的是 "全 5 域多任务 DST"，**不是** "零样本跨域 DST on taxi"
- JGA 数字将被 hotel/restaurant/train 样本稀释，taxi 的真实表现被掩盖
- T5DST / TransferQA 这些 baseline 的对照前提是"只评测 taxi 域"，直接引用其 12-17% 的数字会不对等
- 论文的核心卖点（零样本跨域）将无法被实验数据支撑

**必须决策的三选一**：
- **A. 严格零样本跨域**（推荐，符合 CLAUDE.md）
  - 在 `test.json` 之外再生成一份 `test_taxi.json`，仅保留 belief 中出现 taxi 槽的对话（或仅保留 taxi-only 对话，更严格）
  - evaluator 可选：仅对 taxi 域槽计算 JGA / F1，或整体计算但只报 taxi 数字
  - CLAUDE.md 中 "先行工作残留问题" 的数字（37.54% / 42.25%）才可横向对比
- **B. 改口径为 "任务型 DST 的 CoT 有效性"**
  - 修改 CLAUDE.md 论文定位，去掉 "零样本跨域" 表述
  - 保留当前全域评测，但 baseline 需换成多域 DST 的数字（TRADE 48.62 JGA 等）
- **C. 混合方案**：整体报告 + 按域切片
  - 整体跑 7372 条，但在论文中重点呈现 `domain_slot_accuracy["taxi"]` 作为主指标
  - 此时 `domain_error_breakdown` 已提供必要切片，但 JGA 依然是全域的

**建议**：A 方案。可在预处理后追加一个脚本 `src/filter_taxi_subset.py`，从 `test.json` 筛出 `has_taxi=True` 的子集（≈ 642 条），同时裁剪 gold belief 仅保留 `taxi-*` 槽，evaluator 自然只评 taxi。

---

### 🟠 问题 2：ab_no_s2 / ab_no_s3 的 prompt 文本存在悬空引用

**现状**：`prompt_builder._STEP_TEMPLATES` 中，Step 3 和 Step 4 的文本按编号引用前面步骤：

```
Step 3: "For slots NOT filled by Step 2, check whether..."
Step 4:
  (a) Step 2 produced explicit evidence → fill it.
  (b) Step 3 produced defensible implicit evidence → fill it.
```

当 variant 为 `ab_no_s2` 时（steps=s1+s3+s4），prompt 会说 "对于 Step 2 未填入的槽"，但 **Step 2 根本没出现在上下文里**。`ab_no_s3` 同理。

**影响**：
- 模型可能无法正确理解省略步骤的语义
- 消融结论的可信度被污染：性能下降可能来自"指令不自洽"而非"缺失步骤本身"
- 同行评审时这是明确的方法学漏洞

**修复建议**（两选一）：
- **轻量修**：在 `_STEP_TEMPLATES` 中改写文案，避免硬编码步骤编号。例如 Step 3 改成 "For slots that remain unfilled by explicit evidence, check..."；Step 4 改成 "(a) If explicit evidence was found → fill; (b) If implicit evidence was found → fill; (c) Otherwise → omit"
- **重构**：按保留步骤动态重新编号 S1…Sk，`_make_reasoning_block_header` 已经部分支持此思路，但 `_STEP_TEMPLATES` 里的硬编号需一并改

**优先级**：跑全量之前必改，否则 ablation 结论不可信。

---

### 🟡 问题 3：CLAUDE.md 规划的 "类别型 vs 非类别型" 模板分化未实现

**现状**：`prompt_builder.build_system_prompt` 对所有 slot 使用同一个提示模板，并未区分 `hotel-internet`（yes/no）与 `hotel-name`（自由文本）。CLAUDE.md 中 "模板分两类分别设计" 的设计尚未落地。

**影响**：
- 这是论文方法贡献之一，缺失会削弱方法学完整性
- 但对当前消融实验的"能否跑通"没有阻断性影响

**建议**：
- 若论文需要保留这一贡献：在 `SLOT_SCHEMA` 中为每个 slot 打上 `categorical: True/False` 标签，build_slot_list 时对类别型 slot 显式列出可选值
- 若论文弱化这一贡献：从 CLAUDE.md 中删除相应表述，避免论文与实验脱节

---

### 🟡 问题 4：API 预算与时间评估

**预估规模**（按当前 7372 条全量 × 7 variants）：
- API 调用次数：≈ 51,600
- 单次延迟 0.5s：≈ 7.2 小时纯等待（未计 LLM 实际响应时间，实际约 12–15 小时）
- GPT-4o-mini 费用粗估：输入 ≈ $20，输出 ≈ $14，合计 **≈ $34**（按 input 2.5k / output 450 tokens 估算）

**建议**：
- 先在 500-800 条子样本（比如 `--max_samples 500 --tag pilot500`）上跑 7 组完整流水线，验证对比表结构、错误分类比例是否符合预期，再决定是否全量
- 若采用问题 1 的 A 方案（taxi 子集 ≈ 642 条），实际花费约为估算的 1/10，一小时内即可收工

---

## 四、次要隐患与观察（不阻断实验）

1. **Gold belief 含 schema 外的 `*-booked` 键**（list 结构，来自 MultiWOZ metadata 的预订回执）
   - evaluator.normalize_value 对 non-string 返回 `""` 从而静默过滤，结果正确但不透明
   - 建议在 `download_multiwoz.py::normalize_belief` 中显式 skip `slot == "booked"`

2. **format_history 里 `System:` 行手工 padding 11 个空格**
   - `turn_id ≥ 10` 时对齐会错位，不影响模型输出，仅影响可读性

3. **Slot F1 对 wrong_value 双重计数**（既是 fp 又是 fn）
   - 这是 DST 文献的常见做法，与 T5DST/TransferQA 一致，非 bug

4. **`compare_ablations._render_markdown` 的 Δ 列 `:+d` 格式**
   - 要求输入为 int。目前 error_breakdown 差值确为 int，OK；未来若引入浮点差值需改 `:+.0f`

5. **Gold 端非法 slot 未过滤，pred 端已过滤**（parser._clean_belief 仅清洗 pred）
   - 不对称但不影响数值（gold 端通过 normalize_value 隐式过滤）

6. **`cot_basic` 的 reasoning 要求很弱**（"briefly reason"），可能导致模型忽略推理直接输出 JSON
   - 这恰是"弱 CoT"基线的正确设计，保留即可

7. **没有设置 seed**
   - 由于 temperature=0，GPT-4o-mini 输出确定性，已满足可复现要求

---

## 五、上线前的决策清单（请在此逐条确认）

- [ ] **决策 1**：零样本设定口径 → A / B / C 三选一（见问题 1）
- [ ] **决策 2**：是否修复 ab_no_s2 / ab_no_s3 的悬空引用（强烈建议 ✓）
- [ ] **决策 3**：是否实现 categorical 模板分化（可留到论文 v2）
- [ ] **决策 4**：先 pilot500 再全量，还是直接全量 → 建议 pilot500
- [ ] **决策 5**：tag 命名约定（`taxi_full` / `full_v1` / `ablation_20260423` 等）

---

## 六、推荐的执行顺序

1. 【决策 1 + 修复】确定口径，若选 A：新增 `src/filter_taxi_subset.py` + 改 `TEST_PATH` 指向子集
2. 【修复】改 `prompt_builder._STEP_TEMPLATES`，去掉硬编码步骤编号
3. 【烟测】`python src/export_prompts.py` 导出 7 份 prompt 逐个肉眼过一遍
4. 【烟测】`python src/dst_pipeline.py --mode ablation --suite all --max_samples 200 --tag pilot200` 跑 7 组小规模验证
5. 【检查】查看 `results/ablation_summary_pilot200.md`，确认：
   - cot_full 的 JGA 是否显著高于 standard
   - ab_no_s4 的 hallucinated_slot 是否显著高于 cot_full（论文核心论点）
   - ab_no_s1 的跨域 hallucination 分布是否符合预期
6. 【全量】预期无虞后再跑 `--max_samples 0 --tag full`（或 taxi 子集全量）
7. 【产出】`compare_ablations` 自动生成 `.md` 直接贴论文，`.csv` 喂 matplotlib 画柱状图

---

## 七、当前阶段待办一览

| 序号 | 内容 | 优先级 | 预计用时 |
|---|---|---|---|
| T1 | 与用户确认零样本口径（决策 1） | 🔴 阻断全量 | 5 min 讨论 |
| T2 | 修复 ab_no_s2 / ab_no_s3 prompt 悬空引用 | 🔴 阻断 ablation 可信度 | 15 min |
| T3 | 若选 A 方案：写 `filter_taxi_subset.py` | 🟠 | 20 min |
| T4 | 跑 pilot200 验证 7 组对比表 | 🟠 | 30 min + API |
| T5 | 全量 ablation 实验 | 🟡 | 1h（taxi 子集）/ 12h（全域） |
| T6 | 手工错误分析 100 条（实验四准备） | 🟡 | 阻塞在 T5 之后 |
| T7 | 实现 categorical 模板分化（可选） | ⚪ | 1h，非阻塞 |

---

## 八、修复日志（2026-04-23 第二次会话）

> 本节记录针对第三章四个问题的代码修复结果。修复后未消耗 API 预算；
> 仍需在 pilot200 上验证后再进入全量 ablation。

### 8.1 修复一：prompt 步骤悬空引用（问题 2，🔴）

**改动文件**：`src/prompt_builder.py`

**做法**：
1. 把 `_STEP_TEMPLATES` 中每条模板的标题前缀（"Step 1 — Domain Activation Check"）拆分为两部分：
   - 标题文本仍在模板内（如 "Domain Activation Check:"），但 **去掉了硬编号 "Step N —"**
   - 引入新常量 `_STEP_TITLES` 仅保存短标题（"Domain Activation" 等），用于 `<reasoning>` 块占位符
2. Step 3 / Step 4 文案改用 **概念引用**（"EXPLICIT evidence" / "IMPLICIT evidence"）替代 "Step 2 produced..." / "For slots NOT filled by Step 2..." 这类硬指针
3. 新增 `_format_step_instructions(step_ids)`：按保留的步骤动态从 1 起编号，输出 "Step 1 — ... / Step 2 — ..."
4. `_make_reasoning_block_header` 同步按相同顺序编号

**验证**：
```bash
$ python src/prompt_builder.py            # 7 个 variant 全部成功构建
$ python -c "from prompt_builder import build_system_prompt; print(build_system_prompt('ab_no_s2'))"
# Step 1 = Domain Activation；Step 2 = Implicit Inference；Step 3 = None Verification
# Step 3 的 (a)/(b) 子项引用 EXPLICIT/IMPLICIT 概念，不再悬空指向已删除的 "Step 2"
```

**论文意义**：消融实验的"性能下降"现在只能归因于"缺失的推理能力"，
而不是"指令自相矛盾"——同行评审最忌讳的方法学漏洞已被消除。

---

### 8.2 修复二：零样本跨域评测口径（问题 1，🔴）

采用 **方案 A**（log.md §三推荐方案）。

**新增文件**：`src/filter_taxi_subset.py`
- 入参 `--mode dialogues`（默认）：保留所有 *涉及 taxi* 的对话的全部 1573 轮
  - 与 T5DST / TransferQA 一致：早期未提到 taxi 的轮也在评测内，考验"输出空 taxi 状态"的零样本能力
- 入参 `--mode turns`：更严格，仅保留 belief 含 taxi 槽的 642 轮
- gold belief 在写盘时已裁剪为 `taxi-*` 槽，并显式过滤掉 `taxi-booked` 这类 list 残留
- 输出：`data/processed/test_taxi.json`

**修改文件**：
- `src/utils.py`：新增 `TEST_TAXI_PATH` 与 `resolve_test_path(target_domain)` 工厂函数
- `src/evaluator.py`：`evaluate(...)` / `eval_single(...)` 接受 `target_domain` 参数；新增 `filter_by_domain` 把 pred 也限定为 `taxi-*` 槽（避免模型输出的 hotel-* 被误计为 hallucination）；summary 增加 `target_domain` 字段
- `src/ablation_runner.py`：`run_suite` / `_run_one_variant` 接受 `target_domain`，自动通过 `resolve_test_path` 切换到 `test_taxi.json`，并把 target_domain 传入 evaluator
- `src/dst_pipeline.py`：顶层 CLI 新增 `--target_domain taxi`，三种模式（pilot/single/ablation）全部贯通

**新数据规模**：
| 集合 | turns | dialogs | 含非空 taxi gold 的 turns |
|---|---|---|---|
| `test.json`（全域） | 7372 | 1000 | — |
| `test_taxi.json`（dialogues 模式） | 1573 | 198 | 642 |

**API 成本下修**：1573 turns × 7 variants ≈ **11k 调用**，相比原先 51.6k 降为 ~21%；
按 GPT-4o-mini 估算 **≈ $7**，单次全量 1–2 小时即可收工。

**验证**：用现有 `preds_standard_full.json`（300 条全域 standard 预测）做了一次切片验证：
```
ALL DOMAINS: JGA = 28.0  | Slot_F1 = 79.77 | target_domain = all
TAXI ONLY  : JGA = 92.0  | Slot_F1 = 59.20 | target_domain = taxi
            errors: wrong_value=13, missing_slot=20, hallucinated_slot=5
```
JGA 看起来"虚高"是因为大量 turn 的 taxi gold 为空、模型也没乱填——
这正是论文应当呈现的零样本表现：**模型在源域信息溢出的情况下，能否克制地输出空目标域**。
F1=59.2 才是过滤掉 "all-empty" 配对后的真实槽位水平。

**用法**：
```bash
# 一次性生成 taxi 子集（已运行）
python src/filter_taxi_subset.py

# 后续所有命令加 --target_domain taxi 即自动启用零样本口径
python src/dst_pipeline.py --mode pilot --target_domain taxi
python src/dst_pipeline.py --mode ablation --suite all --target_domain taxi --tag taxi_full
python src/evaluator.py --input results/preds_xxx.json --target_domain taxi
```

---

### 8.3 修复三：`*-booked` list 残留（次要隐患 1）

**改动文件**：`data/download_multiwoz.py` + `src/filter_taxi_subset.py`

- `normalize_belief` 增加显式 `if slot == "booked": continue`，从源头堵掉
  list 结构进入 belief 字典
- `filter_taxi_subset.py` 在裁剪 gold 时也加了 `isinstance(v, str)` 防御，
  保证已生成的 `test_taxi.json` 不含 `taxi-booked` 残留（验证：0 条）

**注意**：现有 `test.json` / `train.json` / `val.json` 是**老版本预处理**的产物，
仍含 `*-booked` list 项。evaluator 通过 `normalize_value` 隐式过滤后结果数值正确，
但若想彻底干净需重跑 `python data/download_multiwoz.py`（约 2 分钟，无 API）。

---

### 8.4 暂未处理的问题

| # | 名称 | 状态 | 决策 |
|---|---|---|---|
| 问题 3 | categorical vs non-categorical 模板分化 | ⚪ 未实现 | 留到论文 v2，当前 ablation 不依赖 |
| 决策 4 | pilot vs 直接全量 | ⚪ 待用户确认 | 推荐 `--max_samples 200 --tag taxi_pilot200` 先验证 |
| 次要隐患 2 | format_history 的 System: 11 空格 padding | ⚪ 未改 | 仅影响可读性，不影响指标 |

---

### 8.5 上线前最终自检清单

执行下面命令应当全部 pass，再开始烧 API：
```bash
python src/prompt_builder.py                        # 7 个 variant 自构建 OK
python src/filter_taxi_subset.py                    # 1573 turns / 198 dialogs
python -c "from src.utils import resolve_test_path; print(resolve_test_path('taxi').name)"
                                                    # → test_taxi.json
python src/evaluator.py --input results/preds_standard_full.json --target_domain taxi
                                                    # JGA / Slot_F1 / target_domain=taxi 三字段齐全
```

下一步：
```bash
python src/dst_pipeline.py --mode ablation --suite all \
    --target_domain taxi --max_samples 200 --tag taxi_pilot200
```
即可在 7 组 × 200 turns ≈ 1400 调用（约 $1）上跑通完整对比表。

---
