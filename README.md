# Zero-Shot DST via Structured Chain-of-Thought

> Bachelor's thesis experiment code — Diagnosing and repairing systematic hallucination and omission in zero-shot Dialogue State Tracking with a four-step structured Chain-of-Thought.

**Paper title (CN)**: 《基于结构化思维链的零样本对话状态追踪研究：面向系统性幻觉与漏判的诊断与修复》

## Research Question

LLMs in zero-shot DST exhibit two systematic gating failures:

- **Hallucinated slots** (false positives) — filling values not in the dialogue
- **Missing slots** (false negatives) — failing to recover implied state

This repo models the problem as a standard **diagnose → localize → patch → verify** engineering loop and proposes a four-step structured CoT, with each step targeting one failure mode.

```
Step 1  Domain activation   → cross-domain spillover hallucinations
Step 2  Explicit search     → value-extraction errors
Step 3  Implicit inference  → omission of implied state
Step 4  None decision       → hallucinated fills (core hypothesis)
```

## Setup

```bash
# Python 3.10+
python -m venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell
pip install -r requirements.txt  # see Phase 0 below

# OpenAI key
$env:OPENAI_API_KEY = "sk-..."

# MultiWOZ 2.1 dataset (not checked in — see data/download_multiwoz.py)
python data/download_multiwoz.py
python src/filter_taxi_subset.py
```

## Experiment Pipeline

| Phase | Command | Cost | Notes |
|---|---|---|---|
| Standard baseline diagnosis | `python src/evaluator.py --input results/preds_standard_full.json --target_domain taxi` | $0 | Phase 1 |
| Pilot ablation (200 × 8) | `python src/dst_pipeline.py --mode ablation --suite all --target_domain taxi --max_samples 200 --tag taxi_pilot200` | ~$1.5 | Phase 2A |
| Full ablation (1573 × 8) | `python src/dst_pipeline.py --mode ablation --suite all --target_domain taxi --tag taxi_full` | ~$8–10 | Phase 2B |
| Comparison table | `python src/compare_ablations.py --tag taxi_full` | $0 | Phase 2B |
| Export prompts | `python src/export_prompts.py` | $0 | sanity check |

See [`process.md`](process.md) for the full phase plan with task dependencies, and [`CLAUDE.md`](CLAUDE.md) for the project narrative and constraints.

## Experiment Conditions (8 variants)

| Variant | Steps retained | Purpose |
|---|---|---|
| `standard` | — | Plain JSON baseline |
| `fn_style` | — | Rich schema, no reasoning — isolates the schema-format contribution |
| `cot_basic` | free-form | Weak CoT baseline (Kojima et al. 2022) |
| `cot_full` | S1+S2+S3+S4 | **Main method** |
| `ab_no_s1` | S2+S3+S4 | Tests Step 1's contribution to cross-domain spillover suppression |
| `ab_no_s2` | S1+S3+S4 | Tests Step 2's contribution to value-extraction accuracy |
| `ab_no_s3` | S1+S2+S4 | Tests Step 3's contribution to implicit-state recovery |
| `ab_no_s4` | S1+S2+S3 | Tests Step 4's contribution to hallucination suppression (core) |

## Dataset

- **MultiWOZ 2.1**, zero-shot cross-domain (source: hotel/restaurant/attraction/train; target: **taxi**)
- Evaluation set: `data/processed/test_taxi.json` (1573 turns / 198 dialogs)
- Model: **GPT-4o-mini** via OpenAI API (no parameter updates)

## Repo Layout

```
src/                  # pipeline source
├── prompt_builder.py    # 8 prompt variants
├── inference.py         # OpenAI calls, resumable
├── parser.py            # JSON extraction + slot validation
├── evaluator.py         # JGA / Slot F1 / error breakdown
├── ablation_runner.py   # multi-variant orchestration
├── compare_ablations.py # MD/CSV/JSON comparison tables
├── dst_pipeline.py      # top-level CLI
├── filter_taxi_subset.py
├── export_prompts.py
└── auto_faithfulness_check.py

results/              # experiment outputs (large preds gitignored)
notebooks/            # faithfulness annotation
prompts/              # exported prompt text for human review
data/                 # MultiWOZ (gitignored — see download script)
relative_researchs/   # reference PDFs (gitignored)
```

## Project Management

Progress is tracked via GitHub Issues + Milestones. See:

- **Milestones** — one per paper chapter (Ch1–Ch6) plus experiment phases (Phase 0–3)
- **Labels** — `phase:0-4`, `type:experiment/analysis/writing`, `priority:P0-P3`
- **Issue templates** — under `.github/ISSUE_TEMPLATE/`

Detailed phase plan: [`process.md`](process.md). Execution log: [`log.md`](log.md).

## License

This is academic work. Code is shared for reproducibility; no warranty.
