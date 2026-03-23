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

## Current Status

`kg-build` is implemented for plain text input:

- reads text file
- runs `kg-gen` extraction
- writes KG artifacts under `data/kg/<kg_name>/`

The extraction currently stores coarse provenance records and JSONL artifacts.

### Dependency Note

This implementation supports running `kg-build` without installing heavy
retrieval extras like `sentence-transformers` or `scikit-learn`.
Those packages are only required if you plan to use `kg-gen` retrieval helpers.
