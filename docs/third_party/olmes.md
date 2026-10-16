# OLMES curated ARC-Challenge examples

`configs/prompts.toml` references the first five entries of
`FEWSHOT_SOURCES["OLMES:ARC-Challenge"]` in upstream OLMES by their
ARC-Challenge `train` split native ids; the shot text is read from the pinned
dataset, not stored in configuration:

- Repository: https://github.com/allenai/olmes
- Commit: `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`
- Path: `oe_eval/tasks/fewshot_sources.py`
- License: Apache-2.0
- Source: [pinned few-shot definitions](https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/tasks/fewshot_sources.py)
- Paper: [OLMES, Figure 5](https://arxiv.org/html/2406.08446v1)

The renderer is implemented locally from the prompt specification.

The upstream Apache license is retained in `olmes-LICENSE` beside this note.
