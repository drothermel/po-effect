# OLMES curated few-shot examples and task formats

`configs/prompts.toml` references the first five entries of each OLMES
few-shot source by split and native id in the pinned dataset; the shot text
is read from the dataset, not stored in configuration. Each referenced row
matches the upstream entry field for field.

| prompt | upstream source | pinned split and native ids |
|---|---|---|
| `olmes_rc`, `olmes_mc`, `cfpo` | `OLMES:ARC-Challenge` | ARC-Challenge `train` ids |
| `olmes_rc_arc_easy` | `OLMES:ARC-Easy` | ARC-Easy `train` ids |
| `olmes_rc_hellaswag` | `OLMES:hellaswag` | `train` row indexes (`ind` repeats within the split) |
| `olmes_rc_piqa` | `OLMES:piqa` | `train` row indexes (no id field) |
| `olmes_rc_winogrande` | `OLMES:winogrande` | `train` row indexes (no id field) |
| `olmes_rc_boolq` | `OLMES:BoolQ` | `train` row indexes (`idx` repeats across splits) |
| `olmes_rc_openbookqa` | `OLMES:openbookqa` | `train` ids |
| `olmes_rc_csqa` | `OLMES:commonsense_qa` | `train` ids |

The query layouts follow each task's `_process_doc` in
`oe_eval/tasks/oe_eval_tasks/`, the evaluation splits follow the
`<task>:rc::olmes` entries in `oe_eval/configs/tasks.py`, and each prompt
entry's provenance names the exact symbols. The renderer is implemented
locally from the prompt specification.

- Repository: https://github.com/allenai/olmes
- Commit: `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`
- Paths: `oe_eval/tasks/fewshot_sources.py`, `oe_eval/tasks/oe_eval_tasks/`,
  `oe_eval/configs/tasks.py`
- License: Apache-2.0
- Source: [pinned few-shot definitions](https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/tasks/fewshot_sources.py)
- Paper: [OLMES, Figure 5](https://arxiv.org/html/2406.08446v1)

The upstream Apache license is retained in `olmes-LICENSE` beside this note.
