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

Converted DataDecide base models and post-trained models with chat templates
are in place. What remains:

- Chat layout variants declared on the prompt's chat block but not built:
  shots as alternating user and assistant turns, and the preamble as a system
  message.
- An API model through OpenRouter, generation only, with the provider's
  reasoning channel recorded as hidden reasoning visibility.

Capabilities stay declared in config until a second model type exists and the
declaration can be checked against a backend.

## Tasks and formulations

- HotpotQA with a two-step solver, the first trajectory attempt kind.
- One generation task with a text judge (exact match and F1).
- Code tasks wait for dr-exec.

## dr-providers needs

Collected during the above and shipped as one release afterward: stop
sequences for generation (po-effect's first-line generation readouts truncate
at the first newline until then), templated scoring and chat template kwargs (po-effect
applies the chat template itself and sends flat text for both scoring and
generation, because the local backend applies a template only on generate,
without kwargs, and its execution evidence names no template), response
template parsing, continuous batching for
local generation, possibly a vLLM backend behind the local backend protocol,
and cross-item batch scoring only if a throughput measurement shows it is
needed.

## Storage

dr-store is integrated: the record store is one SQLite object store per
records root with keyed bindings, and provider evidence is stored there too.
Wall time is no longer part of the attempt record's identity; it is a
separate timing record beside each attempt.

## Optimizers and studies

GEPA runs through whetstone-ai's wrapper around upstream gepa; po-effect
implements whetstone's broker and data-instance protocols over its
measurements and shares one dr-store with it. APE, OPRO, MIPROv2 with Optuna,
and CFPO remain one implementation each in po-effect. Every optimizer treats
a solver spec's prompt slots as the search space and shots as references
into the optimize split. The study layer selects on one
split and reports on another, includes null optimizers, and gates on a
minimum detectable effect.
