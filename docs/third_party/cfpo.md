# CFPO multiple-choice prompt

The `cfpo` prompt in `configs/prompts.toml` copies its instruction, shots
header, and layout from upstream CFPO:

- Repository: https://github.com/microsoft/CFPO
- Commit: `24a720b69d46bf5f07cf097ea61484fefadccbc7`
- License: MIT
- Paper: [CFPO](https://arxiv.org/abs/2502.04295)

What was copied, by path at that commit:

- `src/init_prompts/MultipleChoice.py`: the task instruction text
  (`task_instruction`) and the examples header (`example_hinter`).
- `src/mutators/format_search_pool/query_format/multiple_choice.py`: the
  query layout and choice listing (`QA_renderer_2`).
- `src/mutators/format_search_pool/prompt_renderer/direct_joint.py`: the
  joining of prompt sections (`direct_joint_renderer`).
- `src/prompt.py`: the break after the examples header, the joining of shots,
  and the empty answer on the query (`Prompt.render_examples`,
  `Prompt.render_query`).
- `src/tasks/MultipleChoice.py`: the newline collapse in
  `MultipleChoiceTask.run_evaluate`, which changes none of the pinned rows.

The shots are not from CFPO. They are the ARC-Challenge `train` rows that
OLMES selects (see `olmes.md`), read from the pinned dataset.

The renderer is implemented locally from the prompt specification.

The upstream MIT license is retained in `cfpo-LICENSE` beside this note.
