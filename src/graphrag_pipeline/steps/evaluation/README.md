# evaluation

This step is responsible for benchmark-driven evaluation, with MTRAG as the
planned first benchmark.

## Responsibility

- verify that retrieval and answer outputs can be mapped to MTRAG formats
- adapt stored KG provenance back to passage-level evidence
- run retrieval and generation evaluation scripts

## Planned Benchmark

- `https://github.com/IBM/mt-rag-benchmark`

## Notes

The evaluation path depends on preserving provenance at KG build time.
This module is a scaffold only. No evaluation logic exists yet.
