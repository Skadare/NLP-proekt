# subgraph_retrieval

This step takes the normalized question and retrieves a relevant evidence
subgraph from the canonical KG.

## Responsibility

- accept the normalized question
- locate candidate anchor entities
- score candidate triples or edges
- return a compact subgraph with provenance references

## Input

- normalized question
- `networkx` graph loaded from persisted KG artifacts

## Output

- retrieved facts
- node IDs and triple IDs for the selected subgraph
- provenance references for later answer explanation

This module is a scaffold only. No retrieval logic exists yet.
