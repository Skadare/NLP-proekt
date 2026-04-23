# Pipeline (Simple)

## End-to-end flow

1. **KG build**
   - create benchmark-aligned KG artifacts from MT-RAG passages

2. **Question standardization**
   - rewrite/normalize user query
   - apply alias normalization

3. **Retrieval**
   - retrieve contexts from graph, corpus, or hybrid strategy
   - benchmark retrieval query modes: `lastturn` or `rewrite`

4. **Answer generation (optional)**
   - generate answer from retrieved evidence

5. **Evaluation**
   - retrieval metrics: nDCG and Recall (`@1/@3/@5`)
   - generation evaluation in the same `evaluate` command path

## Minimal run patterns

- Retrieval-only comparison:
  - `... --retrieval-benchmark-mode lastturn --skip-generation --run-eval`
  - `... --retrieval-benchmark-mode rewrite --skip-generation --run-eval`
- Retrieval + generation eval:
  - `... --run-eval` (without `--skip-generation`)

## Near-term plan

- keep `lastturn` and `rewrite` comparisons
- build full-corpus KG (remove 1000-passage cap)
- rerun retrieval and generation eval on full-corpus KG
