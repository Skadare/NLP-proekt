# Architecture

The system is organized around a shared `PipelineContext` that flows through a
sequence of pipeline steps. Each step is isolated in its own module and keeps a
clear contract: read the current context, update it, and return it.

The first version stores KG artifacts on disk as JSONL files and loads them into
`networkx` during retrieval.
