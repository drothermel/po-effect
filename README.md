# po-effect

po-effect compares prompt-optimization methods across tasks and model types
on one fixed evaluation, with one implementation per algorithm. It is designed
from published sources (the OLMES, DataDecide, and CFPO papers and their
upstream repositories) and built on Danielle's `dr-*` infrastructure: dr-hf
for pinned Hub datasets and models, dr-providers for every model call,
dr-serialize for content-addressed identity, and dr-store for records.

## How the pieces fit

A **task** reads rows from a Hugging Face dataset pinned by commit and turns
each into an item with a typed problem half and a typed reference half.
Solvers never see the reference and judges never see the prompt. Item ids are
content-addressed, so a re-pin that changes nothing keeps every id.

A **prompt** is data: a format, a list of shots, and chat settings for
models with a chat template. Shots reference items by split and native id
inside the same pin and task version as the items under evaluation, and
render through the same path as the target item.

A **model handle** declares a pinned checkpoint and the conditions that
change its outputs. Hardware and batching are run conditions, recorded with a
run but not part of any identity. The `po_effect.models` package owns the
dr-providers boundary; everything else sees po-effect types. For a model with
a chat template, that package wraps the rendered prompt as one user message,
applies the template, and records which template and settings it applied.

A **measurement** runs a solver over an item set with a model and stores one
attempt per item: the raw model output and nothing derived from it.
**Readouts** turn stored attempts into candidates and **judges** score them,
as separate records that can be added or replaced without new model calls.
A generation readout first splits the text into reasoning and answer with the
model's response template, which is kept as data.
Every record is a content-addressed document with a named schema and version.
A store behind a single protocol keeps them in one dr-store SQLite database
per records root, under keys that make a measurement's specs, run conditions,
attempts, timings, and readouts addressable; an interrupted run resumes by
skipping items that already have an attempt. Wall time and provider evidence
are stored beside the attempts, outside their identity.

Declarative facts (dataset pins, prompt formats, shot references, model
conditions, response templates) live in TOML under `configs/`, parsed through frozen contracts.

## Working in the repository

```sh
uv sync
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
```

Hub access uses the Hugging Face client's configured authentication and cache.
`scripts/` holds the runners and manual utilities; each prints its own
`--help`.
`scripts/prelim/` holds untested preliminary-sweep tooling driven by jobs
files.
`scripts/prelim/cluster/` holds the Slurm pool tooling that drains a job
queue on the NYU torch cluster; pools read `OPENROUTER_API_KEY` from
`$SCRATCH/.config/secrets/openrouter.env`, which must contain
`export OPENROUTER_API_KEY=...`.

Tests exercise po-effect's own behavior with hand-written rows and a fake
model backend. No test reproduces an upstream artifact, reaches the network,
or compares results against published numbers. Agreement with published
results is checked by hand with a standalone script in `scripts/` that is not
part of the package, has no tests, and is run by no check.

## Design documents

- `docs/design/primitives.py`: a typed sketch of the primitives not yet
  built, written against the types that exist.
- `plans/repo-ecosystem.md`: dependency policy and relationship to the `dr-*`
  repositories.
- `plans/roadmap.md`: remaining work, in order.
- `docs/third_party/`: attribution for copied upstream material.
