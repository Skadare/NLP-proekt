# Important MT-RAG Markdown Files

These are the key files to read for the study paper, in practical order.

## Must read first

- `mt-rag-benchmark/README.md`
  - High-level benchmark overview, corpora sizes, and links to all task/eval docs.

- `mt-rag-benchmark/mtrag-human/README.md`
  - Core human benchmark scope (`110` conversations, `842` tasks), retrieval and generation task settings.

- `mt-rag-benchmark/mtrag-human/retrieval_tasks/README.md`
  - Retrieval task formats (`lastturn`, `rewrite`, `questions`) and baseline retrieval numbers (BM25/BGE/Elser).

- `mt-rag-benchmark/mtrag-human/generation_tasks/README.md`
  - Generation task settings (`reference`, `reference+RAG`, `RAG`) and format details.

- `mt-rag-benchmark/scripts/evaluation/README.md`
  - Official evaluation workflow, required formats, and retrieval/generation evaluation commands.

## Important supporting docs

- `mt-rag-benchmark/corpora/README.md`
  - Passage-level vs document-level usage notes for corpora ingestion.

- `mt-rag-benchmark/mtragun-human/README.md`
  - Next-stage benchmark (`MTRAG-UN`) for UNanswerable, UNderspecified, NONstandalone, and UNclear challenges.

## Optional context docs

- `mt-rag-benchmark/mtrag-synthetic/README.md`
  - Synthetic data details (useful context, lower priority than human benchmark docs).
