# evaluation

This step provides a minimal, course-project-level adapter for the
`mt-rag-benchmark` evaluation scripts. It is intentionally lightweight: format
conversion + a small runner to generate predictions using IBM's official
inputs, checker, and evaluation scripts.

## Responsibility

- map subgraph evidence to passage/chunk evidence for MTRAG `contexts`
- emit retrieval predictions (Task A)
- emit generation predictions (Task B)
- optionally run MTRAG evaluation scripts

## Benchmark

- `https://github.com/IBM/mt-rag-benchmark`

## Provenance Assumption (Explicit)

We assume KG triples carry a `provenance_id` that maps to a
`ProvenanceRecord` with at least one of:

- `passage_id` (preferred)
- `doc_id`
- `source_path`

We also assume `ProvenanceRecord.snippet` is a usable text span for
generation evaluation. If your actual fields differ, update
`mtrag_adapter.py` accordingly.

Note: IBM sample inputs are unrelated to the default `kg_build_sample.txt`,
so retrieval may emit empty `contexts` unless your KG content matches the
IBM questions.

## Custom IBM-Schema Inputs (Aligned to Local KG)

To get meaningful retrieval outputs, use IBM-schema inputs that match your KG.
Two tiny aligned files are provided:

- `src/graphrag_pipeline/steps/evaluation/data_custom/custom_taskac_input.jsonl`
- `src/graphrag_pipeline/steps/evaluation/data_custom/custom_taskb_input.jsonl`

Run with custom inputs:

```bash
python -m graphrag_pipeline.steps.evaluation.runner \
  --kg-dir data/kg/example \
  --mtrag-root /path/to/mt-rag-benchmark \
  --retrieval-input src/graphrag_pipeline/steps/evaluation/data_custom/custom_taskac_input.jsonl \
  --generation-input src/graphrag_pipeline/steps/evaluation/data_custom/custom_taskb_input.jsonl
```

## Minimal Usage (MVP)

This runner reads IBM's official sample inputs and writes prediction files
that match IBM's schema exactly.

```bash
python -m graphrag_pipeline.steps.evaluation.runner \
  --kg-dir data/kg/example \
  --mtrag-root /path/to/mt-rag-benchmark
```

Outputs are written to `data/eval_out` by default:

- `data/eval_out/retrieval_predictions.jsonl`
- `data/eval_out/generation_predictions.jsonl`

To run IBM's official retrieval evaluation script:

```bash
python -m graphrag_pipeline.steps.evaluation.runner \
  --kg-dir data/kg/example \
  --mtrag-root /path/to/mt-rag-benchmark \
  --run-eval
```

## IBM Files and Scripts Used

Inputs (read):

- `/path/to/mt-rag-benchmark/scripts/evaluation/sample_data/taskac_input.jsonl`
- `/path/to/mt-rag-benchmark/scripts/evaluation/sample_data/taskb_input.jsonl`

Checker (called):

- `/path/to/mt-rag-benchmark/scripts/evaluation/format_checker.py`

Evaluation scripts (called when `--run-eval`):

- `/path/to/mt-rag-benchmark/scripts/evaluation/run_retrieval_eval.py`

## Adapter Plug-in Points

- `mtrag_adapter.map_subgraph_to_contexts` populates IBM `contexts`.
- `mtrag_adapter.build_retrieval_record` and `mtrag_adapter.build_generation_record`
  write IBM-schema prediction JSONL objects.
