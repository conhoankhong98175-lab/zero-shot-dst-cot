# Motivation: Cross-Model Error Distribution

> 论文 §5.1 动机验证表。比较 LLM（GPT-4o-mini, standard prompting）与 T5（TransferQA）
> 在零样本跨域 DST 任务中的错误类型分布，验证两类门控错误（幻觉 + 漏判）跨模型成立。

数据来源：
- **GPT-4o-mini**: `results/eval_report_standard_taxi.json`（300 条全域 standard 预测，按 taxi 域投影）
- **TransferQA**: Lin et al. (2021) EMNLP, "Zero-Shot Dialogue State Tracking via Cross-Task Transfer Learning"

---

## 主表：错误类型分布对比

| 错误类型 | GPT-4o-mini (standard, taxi 投影) | TransferQA (T5, MultiWOZ) | 解读 |
|---|---|---|---|
| Hallucinated slot（幻觉填值，假阳性） | **5 / 38 = 13.2%** | **37.54%** | LLM 比 T5 更保守，幻觉更少 |
| Missing slot（隐含漏判，假阴性） | **20 / 38 = 52.6%** | **42.25%** | LLM 漏判压力反而更高 |
| Wrong value（值提取错误） | **13 / 38 = 34.2%** | **20.21%** | LLM 值错占比偏高（多为 taxi-destination 模糊指代） |
| **两类门控错误合计**（幻觉 + 漏判） | **65.8%** | **79.79%** | 跨模型一致：门控错误是主要失效模式 |

---

## 整体指标（仅 GPT-4o-mini，供 §5.2 主表使用）

| 指标 | 数值 |
|---|---|
| 总样本 | 300 |
| JGA（taxi 投影口径） | 92.00% |
| Slot F1 | 59.20% |
| Slot Precision | 67.27% |
| Slot Recall | 52.86% |
| taxi 域 slot accuracy | 52.9% |
| taxi 域跨域 hallucination | 5 |

> 注：JGA 92% 的偏高源于 standard 在 300 条全域样本中大多数对话不涉及 taxi，gold = pred = {} 即算 JGA=True。
> 因此 §5.2 主实验改用 1573 条 taxi 子集（`test_taxi.json`）评测，JGA 数值会显著低于此处。

---

## 关键发现（写入论文 §5.1）

1. **门控错误跨模型成立**：T5（37.54% 幻觉 + 42.25% 漏判 = 79.79%）与 LLM（13.2% + 52.6% = 65.8%）虽然内部比例不同，但两类门控错误合计均占主导，验证本文 CoT 设计动机的普适性。
2. **LLM 的错误倾向反转**：GPT-4o-mini 在零样本 standard 下表现为"过度保守"——更少幻觉、更多漏判。这一发现单独有研究价值，说明：
   - Step 4（None Verification）的边际收益可能在 LLM 上低于 T5
   - Step 3（Implicit Inference）的边际收益可能在 LLM 上高于 T5
   - 消融实验需具体验证（见 §5.3）
3. **Wrong value 集中在 `taxi-destination`**（9/13）：指代消解类错误（"there"/"the museum"），与 Step 2 / Step 3 设计目标对齐。

---

*生成时间：2026-05-14*
