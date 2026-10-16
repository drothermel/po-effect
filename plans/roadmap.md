# Roadmap

Remaining work, in dependency order. Companion to `repo-ecosystem.md`
(dependency strategy) and `docs/design/primitives.py` (sketch of the unbuilt
primitives). Completed work is described by the code and `README.md`, not
here.

## Goal

Full-set measurements of several model types on several tasks with no
optimization in the loop, iterating on the primitives until they stop
changing, then the optimizers and the study layer on top of them.

## Model types

The converted DataDecide base model is in place. Two more model types follow,
each arriving with the smallest task that needs it:

- A small instruct model with a chat template and a response template, so a
  rendered prompt can be a message list and a generation attempt can be split
  into reasoning and answer. The readout owns that split and declares it as
  a transformers response template stored as data.
- An API model through OpenRouter, generation only, with the provider's
  reasoning channel recorded as reasoning visibility on the attempt.

Capabilities stay declared in config until a second model type exists and the
declaration can be checked against a backend.

## Tasks and formulations

- ARC-Challenge in MC form (choices listed, scored over letters) and in the
  CFPO generation form, as additional item formats and readouts over the same
  items, prompt references, and attempt records.
- HotpotQA with a two-step solver, the first trajectory attempt kind.
- One generation task with a text judge (exact match and F1).
- Code tasks wait for dr-exec.

## dr-providers needs

Collected during the above and shipped as one release afterward: response
template parsing, continuous batching for local generation, possibly a vLLM
backend behind the local backend protocol, and cross-item batch scoring only
if a throughput measurement shows it is needed.

## Storage

The file store stays until dr-store is integrated as a second implementation
of the same protocol. The attempt record identity currently includes wall
time; that is removed when records become durable results.

## Optimizers and studies

One implementation per algorithm (APE, OPRO, MIPROv2 with Optuna, GEPA,
CFPO), each treating a solver spec's prompt slots as the search space and
shots as references into the optimize split. The study layer selects on one
split and reports on another, includes null optimizers, and gates on a
minimum detectable effect.
