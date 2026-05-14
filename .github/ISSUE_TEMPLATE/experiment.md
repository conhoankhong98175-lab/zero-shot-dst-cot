---
name: Experiment task
about: A pipeline run, ablation, or evaluation step
title: "[Exp] "
labels: ["type:experiment"]
---

## Goal
What hypothesis is this run testing, or what artifact is being produced?

## Command
```bash
python src/...
```

## Inputs
- Data: `data/processed/...`
- Predecessor task: #

## Expected outputs
- `results/...`

## API cost estimate
~$X.XX (N calls × ~tokens)

## Success criteria
- [ ] Output file exists and parses
- [ ] Metric direction matches design (e.g. `cot_full > standard`)
- [ ] Logged in `log.md`
