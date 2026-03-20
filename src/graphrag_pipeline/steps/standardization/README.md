# standardization

This step normalizes a user question against the canonicalized KG.

## Responsibility

- call an LLM to rewrite the question into a normalized form
- replace aliases with canonical names when appropriate
- preserve the original question in the shared context
- output a normalized question for retrieval

## Input

- raw user question
- KG aliases and canonical entity names

## Output

- `normalized_question`
- linked or candidate canonical mentions

This module is a scaffold only. No normalization logic exists yet.
