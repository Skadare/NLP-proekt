# Benchmark Comparison (dev64)

This compares our latest retrieval runs against the MT-RAG retrieval baseline table in `mt-rag-benchmark/mtrag-human/retrieval_tasks/README.md`.

## Our latest (weighted, all domains)

- Lastturn: R@1 `0.0852`, R@3 `0.1964`, R@5 `0.2190`; nDCG@1 `0.1719`, nDCG@3 `0.1932`, nDCG@5 `0.1980`
- Rewrite: R@1 `0.0969`, R@3 `0.1742`, R@5 `0.2086`; nDCG@1 `0.1719`, nDCG@3 `0.1782`, nDCG@5 `0.1913`

## MT-RAG reference numbers (from benchmark README)

- BM25 lastturn: R@1 `0.08`, R@3 `0.15`, R@5 `0.20`; nDCG@1 `0.17`, nDCG@3 `0.16`, nDCG@5 `0.18`
- BM25 rewrite: R@1 `0.09`, R@3 `0.18`, R@5 `0.25`; nDCG@1 `0.20`, nDCG@3 `0.19`, nDCG@5 `0.22`
- BGE-base 1.5 lastturn: R@1 `0.13`, R@3 `0.24`, R@5 `0.30`; nDCG@1 `0.26`, nDCG@3 `0.25`, nDCG@5 `0.27`
- BGE-base 1.5 rewrite: R@1 `0.17`, R@3 `0.30`, R@5 `0.37`; nDCG@1 `0.34`, nDCG@3 `0.31`, nDCG@5 `0.34`
- Elser lastturn: R@1 `0.18`, R@3 `0.39`, R@5 `0.49`; nDCG@1 `0.42`, nDCG@3 `0.41`, nDCG@5 `0.45`
- Elser rewrite: R@1 `0.20`, R@3 `0.43`, R@5 `0.52`; nDCG@1 `0.46`, nDCG@3 `0.45`, nDCG@5 `0.48`

## Quick comparison

- Against BM25:
  - Lastturn is slightly above BM25 at `@1/@3/@5` for both Recall and nDCG.
  - Rewrite is mixed: above BM25 at `@1`, below BM25 at `@3/@5`.
- Against stronger baselines (BGE/Elser):
  - We are still clearly below on all points.
  - Biggest gap is at deeper ranks (`@3/@5`), especially in rewrite mode.

## Takeaway

- Current best mode for our setup is `lastturn`.
- Current level is roughly BM25-ish to slightly above BM25 in lastturn.
- We are not yet close to BGE/Elser reference levels.

## Important caveat

- Our runs are `dev64` sample runs and use a KG capped to `1000` passages per collection (`data/kg/mtrag_collections_focus`).
- Treat this as directional comparison, not final benchmark positioning.
