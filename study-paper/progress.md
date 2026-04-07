# Progress Log

Short, factual timeline of what we completed.

## Phase 1 - Baseline read and gap check

- Collected prior eval outputs from `data/eval_out/*`.
- Confirmed retrieval aggregates existed; generation predictions existed but no generation metrics were present.
- Mapped benchmark expectations from `mt-rag-benchmark/` docs.

Status: completed.

## Phase 2 - Evaluation path fixes

- Implemented retrieval benchmark modes (`lastturn`, `rewrite`) in evaluator path.
- Fixed generation eval wiring so `--run-eval` can evaluate generation outputs.
- Added judge options (`--judge-provider`, `--judge-model`) with `auto` selection.
- Added explicit full-corpus opt-in for KG build (`--allow-full-corpus`).

Status: completed.

## Phase 3 - Validation of code changes

- Lint checks passed (`ruff check` on changed files).
- Python compile checks passed (`python -m py_compile` on changed files).
- Full pytest run in this environment was blocked earlier by missing dependency (`networkx`) before install.

Status: completed for static checks; environment-dependent tests still need rerun in final runtime setup.

## Phase 4 - Retrieval benchmark runs (latest)

- Ran `dev64` retrieval with notify for:
  - `--retrieval-benchmark-mode lastturn`
  - `--retrieval-benchmark-mode rewrite`
- Both runs passed format checks and produced aggregate metrics.

Key result:

- Lastturn weighted: nDCG@5 `0.19796`, Recall@5 `0.21901`
- Rewrite weighted: nDCG@5 `0.19131`, Recall@5 `0.20860`

Status: completed.

## Phase 5 - Study paper docs

- Updated concise notes in:
  - `study-paper/README.md`
  - `study-paper/project-simple.md`
  - `study-paper/pipeline-simple.md`
  - `study-paper/results-simple.md`
- Added benchmark comparison and key-reading guides:
  - `study-paper/benchmark-comparison.md`
  - `study-paper/important-mtrag-files.md`

Status: completed.

## Current next actions

1. Run generation eval on `dev64` with `--run-eval`.
2. Build full-corpus KG (remove 1000-passages cap) using `--allow-full-corpus`.
3. Rerun retrieval comparison (`lastturn` vs `rewrite`) on full-corpus KG.
4. Re-evaluate generation on full-corpus setup.
