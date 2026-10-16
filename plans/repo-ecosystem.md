# Repo ecosystem and dependency strategy

This file records how po-effect relates to the dr-* repositories and to
external dependencies. The sketch of unbuilt primitives is
`docs/design/primitives.py`.

## Purpose of po-effect

po-effect is the long-term home for systematically comparing prompt
optimization methods across tasks and model types on one fixed evaluation and
one implementation per algorithm. It is designed from published sources
(OLMES, DataDecide, CFPO papers and upstream repos) and from dr-providers, not
from earlier local prototypes.

## Dependency policy

The tradeoff offered by the ecosystem: using dr-* and po-* repos avoids all
dependency hell except what the ecosystem itself introduces.

- Core backbone: the Hugging Face family (transformers, datasets,
  huggingface_hub, tokenizers) plus pydantic. Transitive dependencies of
  these do not count against the policy.
- Small single-purpose libraries are acceptable (httpx, regex, pyarrow,
  numpy, and similar).
- Opinionated frameworks that bring their own model clients, their own task
  vocabulary, or a large pinned dependency tree stay outside the core:
  allenai packages (olmes, ai2-olmo, olmo-core), litellm, DSPy, pydantic-ai,
  inspect-ai, lighteval. Copy and clean the small pieces needed instead.
- Accepted focused libraries, as optional extras:
  - vLLM: an inference engine with one role. Kept as an extra because it
    pins torch and transformers tightly and lags them. It exposes
    `prompt_logprobs`, so it can later serve both generation and
    likelihood scoring behind the dr-providers local backend protocol.
  - Optuna: used only by MIPROv2-style search. Upstream DSPy uses
    `TPESampler(seed=..., multivariate=True)` over categorical choices and
    `optuna.trial.create_trial` to inject externally evaluated points; using
    Optuna keeps that fidelity exact. Runs in memory; sqlalchemy and alembic
    are unused in that mode. Alternatives rejected: hyperopt (univariate
    TPE, dated API), scipy.optimize (no categorical Bayesian search),
    nevergrad (exact-pins bayesian-optimization==1.4.0).
- pydantic-ai is held back until a concrete agentic solver needs it. If
  adopted, it must route model calls through a dr-providers-backed adapter so
  evidence recording stays in one place.
- Copy-and-clean rules: each copied function gets a `third_party` note with
  origin, commit, and license. Copies stay at the level of encode-pair rules,
  text normalizers, F1, and bootstrap, never frameworks.

## Models and datasets live on the Hub

- Only Hugging Face datasets are supported as sources. A new dataset is
  converted to a Hub dataset first; the Hub provides split names, features,
  and immutable commit identity.
- Models that are not cleanly loadable with native transformers classes are
  converted, verified for logit equivalence on a fixed batch, and re-uploaded
  under Danielle's namespace with the original revision naming and a model
  card recording the source repo and commit. The conversion and publishing
  tooling lives in datadec; po-effect consumes the published repositories.
- Measurement records may be stored as parquet via datasets and published
  as Hub datasets, following the DataDecide eval-instances pattern.

## Hugging Face abstractions adopted

- Input: chat templates are the message standard. Reasoning fields are
  model-specific (`reasoning_content`, `thinking`); `continue_final_message`
  accepts a field name; thinking toggles via template kwargs such as
  `enable_thinking`. Base models take raw text. Rendered prompts are either
  raw text or a message list.
- Output: transformers response templates (`tokenizer.response_template`,
  `parse_response(text, prefix=...)`) are the declaration format for
  splitting reasoning, content, and tool calls. po-effect stores its
  response templates as data in `configs/response_templates.toml`, authors
  one when a model ships none, and parses with the transformers parser
  locally. API models expose a single provider-dependent `reasoning` string,
  so readout records carry reasoning visibility.
- Likelihood scoring has no HF abstraction; dr-providers' score call kind is
  the primitive. Transformers continuous batching (`generate_batch`,
  `ContinuousBatchingManager`) is the local generation throughput path and
  belongs in dr-providers, not po-effect.
- Inspect AI's Task/Solver/Scorer vocabulary is the closest whole-pipeline
  standard and lighteval is adopting it. po-effect keeps its own split of
  readout from judge and its first-class likelihood attempt kind, and
  borrows names where they coincide.

## Integration order

External repos are integrated one at a time so each boundary is designed
deliberately. Integrated: dr-providers (all model calls, local and API),
dr-hf (dataset and model pins, all-split reader, named subsets), dr-serialize
(identity documents and hashing), dr-store (records in one SQLite store per
records root), whetstone-ai (GEPA). Remaining, in order of need: dr-exec
(process execution for code judges), dr-graph (multi-step solvers),
dr-platform (cluster execution).

## Boundary with dr-hf

In dr-hf: dataset pin and revision resolution, the all-split reader with
origin split and native id, named subsets keyed by `(name, version)` within a
pin with origin keys derived from the pinned commit, derivations, the
registry and its conflict rule, checkpoint revision parsing, model
publishing with provenance.

In po-effect: task transforms with typed problem and reference halves, item
identity, shot references, prompts, solvers, readouts, judges, measurement
records, optimizers, studies, and all task, prompt, and model configs.

## Existing repos: use, update, replace

Use as-is: dr-serialize, dr-store (integrated; pinned to 0.2.6 through
whetstone-ai), dr-graph, dr-code (code readout and judge via dr-exec).
Use with updates: dr-exec (not a sandbox), dr-platform.
Replace or avoid: dr-llm (no logprobs, superseded), dr-queues (retired),
dr-dspy (DSPy fork; algorithms are reimplemented in po-effect), dr-exp
(training only; reference for platform plus exec wiring).
whetstone-ai: a dependency for GEPA, which runs through its wrapper around
upstream gepa. whetstone-ai and whetstone-envs remain the source of the
study design ideas (select on one split, report on another; null
optimizers; minimum-detectable-effect gate).

## Version policy

Guiding principle, not an automated schedule: every repo in the chain sits
on the latest major version of its external dependencies, and on the latest
minor where practical. Direct dependencies are pinned with a floor and a
major ceiling; exact versions live in the lockfile. When an existing
dependency blocks an upgrade (an exact pin, a lagging engine, an API
removed upstream), the response is to fix the blocker, in our repo or
theirs, rather than to stay on the old version. Upgrades walk leaf-first
(dr-wire, dr-providers, dr-hf, then consumers), each gated by that repo's
conformance tests, with a release per repo. Transformers ships weekly
minors; we do not chase each one.

## Standing policy

The dependency policy, Hub re-upload rule, and version cadence are
ecosystem-wide and are to be recorded in dotfiles as a standing rule for
dr-* and po-* repositories. This file is the po-effect-local statement.
