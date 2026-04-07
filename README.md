# GraphRAG Pipeline Prototype

This repository contains the scaffold for an end-to-end GraphRAG research pipeline built around five stages:

1. `kg_gen` - extract a knowledge graph from text with `kg-gen`
2. `standardization` - normalize user questions against canonical graph aliases
3. `subgraph_retrieval` - retrieve a question-specific evidence subgraph
4. `answering` - generate an answer and a reasoning trace from the subgraph
5. `evaluation` - prepare and evaluate the pipeline with the MTRAG benchmark

## Goals

- Build a modular Python pipeline with shared global context
- Persist knowledge graph artifacts as JSON files and load them into `networkx`
- Expose each stage through a CLI command and a full end-to-end pipeline command
- Keep the initial scaffold implementation-free so each stage can be developed independently

## Project Structure

```text
configs/                 Configuration files for models and pipeline behavior
data/                    Local data, KG artifacts, run outputs, and evaluation assets
docs/                    Architecture and pipeline notes
scripts/                 Helper scripts for dataset/bootstrap tasks
src/graphrag_pipeline/   Main Python package
tests/                   Initial test scaffold
```

## CLI Commands

- `graphrag kg-build --input path/to/file.txt`
- `graphrag kg-build-mtrag --mtrag-root mt-rag-benchmark --output-dir data/kg/<name>`
- `graphrag kg-build-mtrag --mtrag-root mt-rag-benchmark --output-dir data/kg/<name> --source-mode passage-corpus --collection fiqa --max-passages 500`
- `graphrag kg-build-mtrag --mtrag-root mt-rag-benchmark --output-dir data/kg/mtrag_collections --source-mode passage-corpus --split-by-collection --max-passages-per-collection 500 --collection fiqa --collection govt`
- `graphrag kg-build-mtrag --mtrag-root mt-rag-benchmark --output-dir data/kg/mtrag_collections --source-mode passage-corpus --split-by-collection --max-passages-per-collection 500 --progress-every 10 --resume`
- `graphrag kg-build-mtrag --mtrag-root mt-rag-benchmark --output-dir data/kg/mtrag_collections_full --source-mode passage-corpus --split-by-collection --collection clapnq --collection cloud --collection fiqa --collection govt --allow-full-corpus --resume`
- `graphrag normalize --question "..." --kg-dir data/kg/<name>`
- `graphrag retrieve --question "..." --kg-dir data/kg/<name>`
- `graphrag answer --question "..." --kg-dir data/kg/<name>`
- `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/<name>`
- `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/<name> --sample-mode stratified --sample-preset smoke`
- `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/mtrag_collections --sample-mode stratified --sample-preset smoke --top-k 8 --run-eval`
- `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/mtrag_collections --sample-mode stratified --sample-preset dev --top-k 8 --run-eval --progress-every 2 --notify`
- `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/mtrag_collections --sample-mode stratified --sample-preset dev --top-k 8 --retrieval-strategy corpus --run-eval`
- `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/mtrag_collections --sample-mode stratified --sample-preset dev --retrieval-benchmark-mode lastturn --skip-generation --run-eval`
- `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/mtrag_collections --sample-mode stratified --sample-preset dev --retrieval-benchmark-mode rewrite --skip-generation --run-eval`
- `graphrag evaluate --dataset mtrag --mtrag-root mt-rag-benchmark --kg-dir data/kg/mtrag_collections --sample-mode stratified --sample-preset dev --run-eval --judge-provider auto --judge-model ibm-granite/granite-3.3-8b-instruct`
- `graphrag run --question "..." --kg-dir data/kg/<name>`

## Pipeline Design

Each step receives and returns the same shared `PipelineContext` object.
That context carries the current question, normalized form, graph handles,
retrieved subgraph, answer, reasoning, provenance, and generated artifacts.

The pipeline uses a filter-style pattern:

- a runner executes steps in order
- each step reads from the global context
- each step writes its results back to the global context
- the CLI formats the final context into a readable response

## Status

- `kg-build` is implemented with `kg-gen` and persists KG artifacts.
- `kg-build-mtrag` is implemented to preserve MT-RAG passage ids/text in provenance.
- `kg-build-mtrag` supports fast task-context mode and benchmark-faithful passage-corpus mode.
- `kg-build-mtrag` writes `checkpoint.json` files and supports progress logging plus resume for completed collection sub-builds.
- `normalize` is implemented (LLM normalization first, alias replacement second).
- `evaluate` now reads MT-RAG task JSONL directly and emits benchmark-compatible outputs.
- `evaluate` supports stratified sampling presets (`smoke=8`, `dev=64`, `stable=160`).
- `evaluate` routes tasks to per-collection KG subdirectories when available (`clapnq`, `govt`, `fiqa`, `cloud`).
- `evaluate` supports per-phase progress/ETA output and optional desktop notifications (`--notify`).
- `evaluate` supports retrieval strategy selection (`hybrid`, `graph`, `corpus`).
- `evaluate` supports benchmark retrieval query modes (`lastturn`, `rewrite`) and generation judge selection (`auto`, `openai`, `hf`, `vllm`).
