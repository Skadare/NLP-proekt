# standardization

This step normalizes a user question against the canonicalized KG.

## Responsibility

- first call an LLM to rewrite the question into a normalized form
- then replace aliases with canonical names when appropriate
- preserve the original question in the shared context
- output a normalized question for retrieval

## Input

- raw user question
- KG aliases and canonical entity names

## Output

- `normalized_question`
- linked or candidate canonical mentions

## Current Status

The step is implemented with this order:

1. LLM question normalization
2. Alias replacement from `aliases.jsonl` in `kg_dir`

If LLM normalization fails (for example missing API key), the step falls back
to the original question and still runs alias replacement.
