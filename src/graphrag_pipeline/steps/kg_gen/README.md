# kg_gen

This step converts raw text into a persisted knowledge graph using `kg-gen`.

## Responsibility

- accept a path to a text file
- run KG extraction with `kg-gen`
- save entities, relations, triples, aliases, and provenance as JSON artifacts
- prepare graph data that can later be loaded into `networkx`

## Planned Command

`graphrag kg-build --input path/to/file.txt`

## Outputs

- `entities.jsonl`
- `relations.jsonl`
- `triples.jsonl`
- `aliases.jsonl`
- `provenance.jsonl`
- `metadata.json`

This module is a scaffold only. No extraction logic exists yet.
