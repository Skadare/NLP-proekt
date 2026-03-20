# answering

This step converts a retrieved subgraph into a user-facing answer and a readable
reasoning trace.

## Responsibility

- first LLM call: answer from question + subgraph
- second LLM call: generate reasoning from question + subgraph + answer + provenance
- keep the final output readable and evidence-aware

## Input

- raw or normalized question
- retrieved subgraph
- provenance records for the selected evidence

## Output

- `answer`
- `reasoning`
- evidence identifiers used during generation

This module is a scaffold only. No answer generation logic exists yet.
