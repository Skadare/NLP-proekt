# Project (Simple)

## What this project is

Early-stage GraphRAG pipeline in Python.

Main workflow:

1. Build MT-RAG-aligned KG artifacts (`kg-build-mtrag`)
2. Normalize question text
3. Retrieve evidence (graph/corpus/hybrid)
4. Optionally generate an answer
5. Evaluate retrieval and generation (`evaluate`)

## Useful commands

- Build benchmark KG:
  - `graphrag kg-build-mtrag --mtrag-root mt-rag-benchmark --output-dir data/kg/<name> ...`
- Compare retrieval query modes:
  - `--retrieval-benchmark-mode lastturn`
  - `--retrieval-benchmark-mode rewrite`
- Run retrieval + generation eval:
  - `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/<name> --run-eval`

## Current status

- Prototype is working and benchmark-compatible.
- Current focused KG (`data/kg/mtrag_collections_focus`) is capped to `1000` passages per collection.
- Next milestone is full-corpus KG build and rerun of retrieval + generation evaluation.
