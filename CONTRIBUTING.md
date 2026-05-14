# Contributing / Working on this repo

Solo thesis project, but conventions matter for using parallel work tools cleanly.

## Branch naming

| Prefix | Use for | Example |
|---|---|---|
| `feat/` | Code changes (pipeline, prompt, evaluator) | `feat/fn-style`, `feat/auto-faithfulness` |
| `exp/` | Experiment-run branches (rarely merged) | `exp/taxi-pilot200` |
| `writing/` | Paper chapter drafts | `writing/ch2-related-work`, `writing/ch5-section3` |
| `analysis/` | Analysis artifacts (csv/md/notebooks) | `analysis/step-error-mapping` |
| `fix/` | Bug fixes | `fix/evaluator-taxi-projection` |

PRs reference the issue they close: `Closes #12`. Merging the PR auto-closes the issue and shows up in the milestone view.

## Parallel work via `git worktree`

Same `.git/`, multiple working trees. Lets a long experiment run on `main` without blocking writing or prompt changes.

```powershell
# From the main repo directory
git worktree add ../project-ch2 writing/ch2-related-work
git worktree list

# When done
git worktree remove ../project-ch2
```

Pre-configured worktrees in this repo:

| Path | Branch | Intended task |
|---|---|---|
| `../project-fn-style` | `feat/fn-style` | T0.1 — add fn_style variant |
| `../project-ch2` | `writing/ch2-related-work` | T4.1 — chapter 2 draft |
| `../project-ch3` | `writing/ch3-methodology` | T4.2 — chapter 3 draft |
| `../project-sandbox` | `analysis/sandbox` | throwaway scripts / experiments |

### Python env per worktree

Each worktree has its own `.venv/` (gitignored). Bootstrap once per worktree:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Copy your OPENAI_API_KEY into .env (or set $env:OPENAI_API_KEY)
copy ..\project\.env .env   # if .env already exists in main
```

Alternative (lighter): activate the main repo's `.venv` from inside a worktree — works as long as Python version doesn't change.

## Long-running experiments

Phase 2B (`tag=taxi_full`) takes 1.5–2.5h. Recommended pattern:

1. Run on `main` (or a dedicated `exp/` branch worktree)
2. Use `run_in_background` if invoking via Claude Code, or `Start-Process` / tmux locally
3. Pipeline supports resume — re-running the same command picks up where it left off (key: `dial_id + turn_id`)

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`:

- `ruff check src/` — lint
- `python -m compileall src` — syntax
- import smoke test on top-level pipeline modules

Does **not** call the OpenAI API. Experiment runs stay manual.

## Local sanity check before pushing

```powershell
pip install ruff
ruff check src/
python -m compileall -q src
```

## Issue / milestone discipline

- Each task in `process.md` has a tracking issue. When you finish work, close via PR (`Closes #N`) — the milestone progress bar moves on its own.
- New unplanned work → open an issue with the right `phase:*` + `type:*` + `priority:*` labels before starting. The 4 templates under `.github/ISSUE_TEMPLATE/` cover the common shapes.
