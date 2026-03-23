# AGENTS.md
Guidance for coding agents working in this repository.
This project is an early-stage GraphRAG scaffold written in Python.
## 1) Repository Snapshot
- Package: `graphrag-pipeline`
- Source root: `src/graphrag_pipeline`
- Test root: `tests`
- CLI: `graphrag` -> `graphrag_pipeline.cli:app`
- Python: `>=3.11`
- Core deps: `networkx`, `pydantic`, `typer`, `rich`
- Dev deps: `pytest`, `ruff`
## 2) Cursor/Copilot Rules Status
Checked:
- `.cursorrules`
- `.cursor/rules/`
- `.github/copilot-instructions.md`
Current state:
- No Cursor or Copilot instruction files are present.
- If these files are added later, treat them as authoritative and sync this file.
## 3) Setup
From repo root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```
If CLI is unavailable, run:
```bash
python -m graphrag_pipeline.cli --help
```
## 4) Build / Run / Lint / Test Commands
Install editable package:
```bash
pip install -e '.[dev]'
```
Run CLI help:
```bash
graphrag --help
```
Run scaffold pipeline command:
```bash
graphrag run --question "What is GraphRAG?" --kg-dir data/kg/example
```
Note: many commands currently raise `NotImplementedError` intentionally.
Lint:
```bash
ruff check .
```
Format check:
```bash
ruff format --check .
```
Format files:
```bash
ruff format .
```
Run all tests:
```bash
pytest
```
Run one test file:
```bash
pytest tests/test_pipeline_runner.py
```
Run one test function:
```bash
pytest tests/test_pipeline_runner.py::test_default_runner_preserves_question
```
Run tests by substring:
```bash
pytest -k context
```
## 5) Architecture Expectations
- Keep the filter-style pipeline pattern.
- Each step must accept and return `PipelineContext`.
- Use global context as the integration boundary between steps.
- Prefer extending typed models over passing loose dictionaries.
- Keep CLI output structured and readable.
Primary step packages:
- `steps/kg_gen`
- `steps/standardization`
- `steps/subgraph_retrieval`
- `steps/answering`
- `steps/evaluation`
## 6) Code Style Guidelines
### Imports
- Group imports: stdlib, third-party, local.
- Use absolute imports for cross-package references when clear.
- Use relative imports only for nearby modules if consistent.
- Remove unused imports.
- Avoid wildcard imports.
### Formatting
- Follow Ruff defaults.
- Respect line length `100`.
- Add concise docstrings for modules/classes/functions.
- Keep functions focused on one responsibility.
- Keep CLI handlers thin; move logic to step/service modules.
### Types and Models
- Add type hints to all public APIs.
- Use `pydantic.BaseModel` for shared contracts.
- Use explicit optional types (`str | None`).
- Use `Field(default_factory=...)` for mutable defaults.
- Prefer explicit domain names (`Triple`, `ProvenanceRecord`, `SubgraphResult`).
### Naming
- Modules/packages: `snake_case`
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- CLI commands: short, explicit, verb-oriented.
### Error Handling
- Never silently swallow exceptions.
- Raise specific exceptions with actionable messages.
- Include context in errors (path, ids, command args).
- Fail fast for invalid CLI inputs.
- Convert low-level exceptions to user-facing errors near CLI boundaries.
### Testing
- Add or update tests for non-trivial changes.
- Keep tests deterministic and focused.
- For pipeline steps, test happy and failure paths.
- Use fixtures for reusable context/graph setup.
## 7) Data and Artifact Rules
- Treat `data/` as local artifacts (gitignored).
- Persist KG artifacts in machine-readable formats (`jsonl`, metadata files).
- Preserve provenance fields through all transformations.
- Never commit secrets; keep credentials in `.env`.
## 8) Agent Workflow Rules
- Inspect related modules before editing.
- Keep changes minimal and scoped.
- Avoid unrelated refactors in the same patch.
- Update docs when behavior/commands change.
- If adding commands, document usage in `README.md` and here.
## 9) Known Constraints
- Many modules are placeholders by design.
- Several CLI commands intentionally raise `NotImplementedError`.
- Dependencies may be missing until install is run.
- MTRAG integration is planned but not implemented.
## 10) Definition of Done
- Changed code imports/executes in the target environment.
- Ruff passes for changed scope (or repo when feasible).
- Tests pass for affected areas.
- Relevant README/docs are updated.
- No secrets or local data artifacts are committed.
