# Results (Simple)

## Latest signal (dev64, `data/kg/mtrag_collections_focus`)

Overall weighted retrieval:

- `lastturn`: nDCG@5 `0.19796`, Recall@5 `0.21901`
- `rewrite`: nDCG@5 `0.19131`, Recall@5 `0.20860`

Current takeaway: `lastturn` is better overall on this setup.

## Domain pattern

`lastturn` is better on:

- `clapnq`
- `govt`

`rewrite` is slightly better on:

- `cloud`
- `fiqa`

## Caveats

- This is a `dev` sample (64 tasks), not a full benchmark run.
- KG is currently capped to `1000` passages per collection.

## Next steps

1. Build full-corpus KG (remove per-collection cap).
2. Rerun retrieval comparison: `lastturn` vs `rewrite`.
3. Run generation evaluation in the same rerun.
4. Re-check domain behavior after full-corpus indexing.
