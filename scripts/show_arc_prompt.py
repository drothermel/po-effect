from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dr_hf import read_rows
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from po_effect.config import DEFAULT_CONFIG_DIR, load_config
from po_effect.measurement import (
    Abstention,
    Candidate,
    DecisionRule,
    GenerationAttempt,
    GenerationSolver,
    GenerationSpec,
    IndexCandidate,
    IndexMatchJudge,
    LikelihoodArgmaxReadout,
    ReadoutId,
    ScoredCandidatesAttempt,
    ScoredCandidatesSolver,
    SingleCallSpec,
    readout_for,
    readout_ids_for,
    split_response,
    truncated_text,
)
from po_effect.models import (
    ChatTemplate,
    Device,
    ModelHandle,
    ModelsConfig,
    ModelSession,
    ResponseTemplate,
    RunConditions,
    chat_template_for,
    has_chat_template,
    load_model_handle,
    load_models_config,
    load_response_template,
    load_tokenizer,
)
from po_effect.prompts import (
    ChatSettings,
    MultipleChoiceRenderer,
    PromptsConfig,
    PromptSpec,
    load_prompt_config,
    resolve_prompt,
)
from po_effect.tasks import (
    ArcChallengeTask,
    AttemptKind,
    Item,
    ItemIndex,
    ItemSet,
    ItemSetDerivation,
    ItemSetRole,
    build_item_index,
    load_arc_challenge,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_PROMPT = "olmes_rc"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 16


def prompt_names(config_dir: Path) -> list[str]:
    return sorted(
        load_config(config_dir / "prompts.toml", PromptsConfig).prompts
    )


SIZE_TOKEN = re.compile(r"\d+[bm]")
DECIMAL_TOKEN = re.compile(r"\d")


class LineBreakHelpFormatter(argparse.HelpFormatter):
    def _split_lines(self, text: str, width: int) -> list[str]:
        wrapped: list[str] = []
        for line in text.splitlines():
            wrapped.extend(super()._split_lines(line, width))
        return wrapped


def size_prefix(name: str) -> tuple[str, ...]:
    tokens = name.split("_")
    for position, token in enumerate(tokens):
        if SIZE_TOKEN.fullmatch(token):
            return tuple(tokens[:position])
    return tuple(tokens)


def model_families(names: list[str]) -> dict[str, list[str]]:
    prefixes = {size_prefix(name) for name in names}
    families: dict[str, list[str]] = {}
    for name in names:
        prefix = size_prefix(name)
        if (
            prefix
            and DECIMAL_TOKEN.fullmatch(prefix[-1])
            and prefix[:-1] in prefixes
        ):
            prefix = prefix[:-1]
        families.setdefault("_".join(prefix), []).append(name)
    return families


def model_help(models: ModelsConfig) -> str:
    lines = ["model entry from models.toml for --run:"]
    for family, names in model_families(list(models.models)).items():
        labels = [
            f"{name} (default)" if name == models.default else name
            for name in names
        ]
        lines.append(f"{family}: {', '.join(labels)}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    config = argparse.ArgumentParser(add_help=False)
    config.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    config_dir = config.parse_known_args(argv)[0].config_dir
    names = prompt_names(config_dir)
    models = load_models_config(config_dir)
    parser = argparse.ArgumentParser(
        parents=[config],
        formatter_class=LineBreakHelpFormatter,
        description=(
            "Print the exact context, candidate continuations, and "
            "unconditional context sent to the model for one "
            "ARC-Challenge item under the selected prompt configuration."
        ),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--native-id")
    selection.add_argument("--index", type=int)
    parser.add_argument(
        "--format",
        default=DEFAULT_PROMPT,
        choices=names,
        help=f"prompt entry from prompts.toml: {', '.join(names)}",
    )
    parser.add_argument("--show-whitespace", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "run the model on the item in the prompt entry's declared "
            "attempt kind: score prints the candidate scores and "
            "per-decision-rule readouts, generate prints the raw and "
            "truncated text, the applicable readout, and the judge's "
            "scores; requires --device"
        ),
    )
    parser.add_argument(
        "--attempt",
        choices=[kind.value for kind in AttemptKind],
        help=(
            "override the prompt entry's declared attempt kind; requires "
            "--run and must be admitted by the entry"
        ),
    )
    parser.add_argument(
        "--device",
        choices=[device.value for device in Device],
        help="device for --run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(f"batch size for --run (default {DEFAULT_BATCH_SIZE})"),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help=(
            "token limit when --run generates "
            f"(default {DEFAULT_MAX_NEW_TOKENS}); ignored when it scores"
        ),
    )
    parser.add_argument(
        "--model",
        choices=sorted(models.models),
        metavar="NAME",
        help=model_help(models),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "override the prompt entry's chat block with enable_thinking "
            "true; requires a model with the chat_template capability"
        ),
    )
    args = parser.parse_args(argv)
    if args.run and args.device is None:
        parser.error("--run requires --device")
    if args.attempt is not None and not args.run:
        parser.error("--attempt requires --run")
    if args.model is None:
        args.model = models.default
    if args.enable_thinking and not has_chat_template(
        models.models[args.model]
    ):
        parser.error(
            f"--enable-thinking requires the chat_template capability, "
            f"which {args.model} lacks"
        )
    return args


@dataclass(frozen=True, slots=True)
class RunModel:
    handle: ModelHandle
    tokenizer: PreTrainedTokenizerBase | None
    response_template: ResponseTemplate | None


def thinking_prompt(prompt: PromptSpec) -> PromptSpec:
    chat = (
        ChatSettings(
            enable_thinking=True,
            shots_as_turns=False,
            preamble_as_system=False,
        )
        if prompt.chat is None
        else prompt.chat.model_copy(update={"enable_thinking": True})
    )
    return prompt.model_copy(update={"chat": chat})


def template_label(template: ChatTemplate) -> Text:
    applied = template.applied
    return Text(
        f"chat template sha256 {applied.template_sha256[:12]}, "
        f"kwargs {applied.template_kwargs!r}"
    )


def evaluation_item_set(task: ArcChallengeTask, items: ItemIndex) -> ItemSet:
    origin = items.splits.get(task.config.split)
    if origin is None:
        raise ValueError(f"unknown evaluation split {task.config.split!r}")
    return ItemSet(
        name=origin.key.name,
        task_id=task.task_id,
        task_version=task.version,
        role=ItemSetRole.REPORT,
        item_ids=tuple(items.item_ids[key] for key in origin.native_ids),
        derivation=ItemSetDerivation(
            method="origin_split", parent=origin.content_hash
        ),
    )


def select_item(
    args: argparse.Namespace, item_set: ItemSet, items: ItemIndex
) -> Item:
    if args.native_id is not None:
        item_id = items.item_ids.get(args.native_id)
        if item_id is None or item_id not in item_set.item_ids:
            raise SystemExit(
                f"native id {args.native_id!r} is not in the evaluation set"
            )
        return items.get(item_id)
    index = 0 if args.index is None else args.index
    if not 0 <= index < len(item_set.item_ids):
        raise SystemExit(
            f"index {index} is outside the evaluation set of "
            f"{len(item_set.item_ids)} items"
        )
    return items.get(item_set.item_ids[index])


def exact(value: str, *, show_whitespace: bool) -> Text:
    shown = repr(value) if show_whitespace else value
    return Text(shown, overflow="fold", end="")


def step_rule(console: Console, number: int, title: str) -> None:
    console.rule(Text(f"{number}. {title}"), align="left")


def optional_float(value: float | None) -> Text:
    return Text("" if value is None else repr(value))


def gold_line(item: Item) -> Text:
    return Text(f"gold index {item.reference.gold_index}")


def model_line(args: argparse.Namespace, *extra: str) -> Text:
    return Text(
        ", ".join((f"model {args.model}", f"device {args.device}", *extra))
    )


def raw_score_table(attempt: ScoredCandidatesAttempt) -> Table:
    table = Table(title=Text("raw candidate scores"), highlight=False)
    table.add_column("index", justify="right")
    table.add_column("loglik", justify="right", no_wrap=True)
    table.add_column("tokens", justify="right")
    table.add_column("uncond loglik", justify="right", no_wrap=True)
    for position, score in enumerate(attempt.scores):
        table.add_row(
            Text(str(position)),
            optional_float(score.log_likelihood),
            Text(str(score.token_count)),
            optional_float(score.unconditional_log_likelihood),
        )
    return table


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    rule: DecisionRule
    candidate: IndexCandidate | None = None
    inapplicable: str | None = None
    error: str | None = None


def rule_outcomes(
    attempt: ScoredCandidatesAttempt, item: Item, prompt: PromptSpec
) -> list[RuleOutcome]:
    outcomes: list[RuleOutcome] = []
    for rule in DecisionRule:
        readout = LikelihoodArgmaxReadout(rule)
        reason = readout.inapplicable_reason(
            attempt.kind, prompt.item_format.candidates, (attempt,)
        )
        if reason is not None:
            outcomes.append(RuleOutcome(rule, inapplicable=reason))
            continue
        try:
            candidate = readout.read(attempt, item.problem)
        except (ValueError, ZeroDivisionError) as error:
            outcomes.append(
                RuleOutcome(rule, error=f"readout cannot apply: {error}")
            )
            continue
        outcomes.append(RuleOutcome(rule, candidate=candidate))
    return outcomes


def readout_table(
    attempt: ScoredCandidatesAttempt,
    outcomes: list[RuleOutcome],
    *,
    show_whitespace: bool,
) -> Table:
    table = Table(title=Text("decision rules"), highlight=False)
    table.add_column("rule")
    table.add_column("chosen", justify="right")
    table.add_column("continuation", overflow="fold")
    for outcome in outcomes:
        rule = Text(outcome.rule.value)
        if outcome.inapplicable is not None:
            table.add_row(
                rule, Text("not applicable"), Text(outcome.inapplicable)
            )
        elif outcome.candidate is None:
            table.add_row(rule, Text(""), Text(outcome.error or ""))
        else:
            index = outcome.candidate.index
            table.add_row(
                rule,
                Text(str(index)),
                exact(
                    attempt.candidates[index], show_whitespace=show_whitespace
                ),
            )
    return table


def score_evaluation_table(outcomes: list[RuleOutcome], item: Item) -> Table:
    table = Table(title=Text("judge results"), highlight=False)
    table.add_column("rule")
    table.add_column("chosen", justify="right")
    table.add_column("matches gold")
    judge = IndexMatchJudge()
    for outcome in outcomes:
        if outcome.inapplicable is not None:
            continue
        if outcome.candidate is None:
            table.add_row(Text(outcome.rule.value), Text(""), Text(""))
            continue
        verdict = judge.judge(outcome.candidate, item.reference)
        table.add_row(
            Text(outcome.rule.value),
            Text(str(outcome.candidate.index)),
            Text("yes" if verdict.accuracy == 1 else "no"),
        )
    return table


def score_item(  # noqa: PLR0913, PLR0917
    console: Console,
    args: argparse.Namespace,
    model: RunModel,
    prompt: PromptSpec,
    item: Item,
    items: ItemIndex,
) -> None:
    solver = ScoredCandidatesSolver(MultipleChoiceRenderer())
    spec = SingleCallSpec(prompt=prompt)
    with open_session(args, model, prompt) as session:
        attempt = solver.solve(item, session, spec, items)
    show = args.show_whitespace
    step_rule(console, 2, "Model output")
    console.print(model_line(args))
    if args.max_new_tokens is not None:
        console.print(
            Text("--max-new-tokens ignored: score attempts do not generate")
        )
    console.print(raw_score_table(attempt))
    outcomes = rule_outcomes(attempt, item, prompt)
    step_rule(console, 3, "Readout")
    console.print(readout_table(attempt, outcomes, show_whitespace=show))
    step_rule(console, 4, "Evaluation")
    console.print(gold_line(item))
    console.print(score_evaluation_table(outcomes, item))


def open_session(
    args: argparse.Namespace, model: RunModel, prompt: PromptSpec
) -> ModelSession:
    return ModelSession(
        model.handle,
        RunConditions(device=Device(args.device), batch_size=args.batch_size),
        prompt.chat,
        tokenizer=model.tokenizer,
    )


def raw_generation_panel(
    attempt: GenerationAttempt, *, show_whitespace: bool
) -> Panel:
    output = attempt.output
    return Panel(
        exact(output.text, show_whitespace=show_whitespace),
        title=Text("generated text"),
        title_align="left",
        subtitle=Text(
            f"stop reason {output.stop_reason}, "
            f"prompt tokens {output.prompt_tokens}, "
            f"completion tokens {output.completion_tokens}"
        ),
        subtitle_align="left",
    )


def split_panels(
    attempt: GenerationAttempt,
    template: ResponseTemplate | None,
    *,
    show_whitespace: bool,
) -> tuple[Panel, Panel, Panel]:
    split = split_response(attempt.output.text, template)
    reasoning = Panel(
        Text("none")
        if split.reasoning is None
        else exact(split.reasoning, show_whitespace=show_whitespace),
        title=Text("reasoning"),
        title_align="left",
        subtitle=Text(
            f"reasoning visibility {split.reasoning_visibility.value}, "
            + (
                "no response template"
                if template is None
                else "split by the response template"
            )
        ),
        subtitle_align="left",
    )
    answer = Panel(
        exact(split.answer, show_whitespace=show_whitespace),
        title=Text("answer"),
        title_align="left",
    )
    truncated = Panel(
        exact(truncated_text(split.answer), show_whitespace=show_whitespace),
        title=Text("answer truncated at the first newline"),
        title_align="left",
    )
    return reasoning, answer, truncated


GenerationOutcome = tuple[ReadoutId, Candidate | Abstention]


def generation_outcomes(
    attempt: GenerationAttempt,
    item: Item,
    prompt: PromptSpec,
    template: ResponseTemplate | None,
) -> list[GenerationOutcome]:
    outcomes: list[GenerationOutcome] = []
    candidates = prompt.item_format.candidates
    for readout_id in readout_ids_for(AttemptKind.GENERATE):
        readout = readout_for(readout_id, template)
        reason = readout.inapplicable_reason(
            attempt.kind, candidates, (attempt,)
        )
        if reason is not None:
            continue
        outcomes.append((readout_id, readout.read(attempt, item.problem)))
    return outcomes


def generation_readout_table(
    outcomes: list[GenerationOutcome], item: Item, *, show_whitespace: bool
) -> Table:
    table = Table(title=Text("generation readout"), highlight=False)
    table.add_column("readout")
    table.add_column("candidate", justify="right")
    table.add_column("choice or abstention reason", overflow="fold")
    for readout_id, candidate in outcomes:
        if isinstance(candidate, Abstention):
            chosen = Text("abstained")
            detail = Text(candidate.reason)
        else:
            chosen = Text(str(candidate.index))
            detail = exact(
                item.problem.choices[candidate.index],
                show_whitespace=show_whitespace,
            )
        table.add_row(Text(readout_id.value), chosen, detail)
    return table


def generation_evaluation_table(
    outcomes: list[GenerationOutcome], item: Item
) -> Table:
    table = Table(title=Text("judge results"), highlight=False)
    table.add_column("readout")
    table.add_column("accuracy", justify="right")
    table.add_column("abstained", justify="right")
    judge = IndexMatchJudge()
    for readout_id, candidate in outcomes:
        scores = judge.judge(candidate, item.reference)
        table.add_row(
            Text(readout_id.value),
            Text(f"{scores.accuracy:g}"),
            Text(str(scores.abstained)),
        )
    return table


def generate_item(  # noqa: PLR0913, PLR0917
    console: Console,
    args: argparse.Namespace,
    model: RunModel,
    prompt: PromptSpec,
    item: Item,
    items: ItemIndex,
) -> None:
    solver = GenerationSolver(MultipleChoiceRenderer())
    max_new_tokens = (
        DEFAULT_MAX_NEW_TOKENS
        if args.max_new_tokens is None
        else args.max_new_tokens
    )
    spec = GenerationSpec(prompt=prompt, max_new_tokens=max_new_tokens)
    with open_session(args, model, prompt) as session:
        attempt = solver.solve(item, session, spec, items)
    show = args.show_whitespace
    step_rule(console, 2, "Model output")
    console.print(model_line(args, f"max new tokens {max_new_tokens}"))
    console.print(raw_generation_panel(attempt, show_whitespace=show))
    template = model.response_template
    outcomes = generation_outcomes(attempt, item, prompt, template)
    step_rule(console, 3, "Readout")
    for panel in split_panels(attempt, template, show_whitespace=show):
        console.print(panel)
    console.print(
        generation_readout_table(outcomes, item, show_whitespace=show)
    )
    step_rule(console, 4, "Evaluation")
    console.print(gold_line(item))
    console.print(generation_evaluation_table(outcomes, item))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = load_arc_challenge(args.config_dir)
    prompt_config = load_prompt_config(args.config_dir, args.format)
    items = build_item_index(task, tuple(read_rows(task.source)))
    item_set = evaluation_item_set(task, items)
    prompt = resolve_prompt(args.format, prompt_config, task, items)
    if args.enable_thinking:
        prompt = thinking_prompt(prompt)
    handle = load_model_handle(args.config_dir, args.model)
    named = load_response_template(args.config_dir, handle.response_template)
    model = RunModel(
        handle=handle,
        tokenizer=load_tokenizer(handle)
        if has_chat_template(handle)
        else None,
        response_template=None if named is None else named.template,
    )
    chat_template = chat_template_for(handle, prompt.chat, model.tokenizer)
    kind = (
        prompt.attempt if args.attempt is None else AttemptKind(args.attempt)
    )
    if args.run:
        try:
            prompt.require_admitted(kind)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    item = select_item(args, item_set, items)
    rendered = MultipleChoiceRenderer().render(
        prompt,
        item.problem,
        tuple(items.get(i) for i in prompt.shot_item_ids),
    )
    show = args.show_whitespace
    console = Console(highlight=False)
    step_rule(
        console,
        1,
        f"Prompt: {item.native_id} ({args.format}, {kind.value} attempt)",
    )
    console.print(
        Panel(
            exact(rendered.context, show_whitespace=show),
            title=Text(item.native_id),
            title_align="left",
        )
    )
    if chat_template is not None:
        console.print(
            Panel(
                exact(
                    chat_template.apply(rendered.context),
                    show_whitespace=show,
                ),
                title=Text(f"templated context sent to {args.model}"),
                title_align="left",
                subtitle=template_label(chat_template),
                subtitle_align="left",
            )
        )
    table = Table(title=Text("candidates"), highlight=False)
    table.add_column("index", justify="right")
    table.add_column("continuation", overflow="fold")
    table.add_column("gold")
    for position, candidate in enumerate(rendered.candidates):
        table.add_row(
            Text(str(position)),
            exact(candidate, show_whitespace=show),
            Text("*" if position == item.reference.gold_index else ""),
        )
    console.print(table)
    unconditional = rendered.unconditional_context
    console.print(
        Panel(
            Text("none")
            if unconditional is None
            else exact(unconditional, show_whitespace=show),
            title=Text("unconditional context"),
            title_align="left",
        )
    )
    if chat_template is not None and unconditional is not None:
        console.print(
            Panel(
                exact(
                    chat_template.apply(unconditional),
                    show_whitespace=show,
                ),
                title=Text(
                    f"templated unconditional context sent to {args.model}"
                ),
                title_align="left",
                subtitle=template_label(chat_template),
                subtitle_align="left",
            )
        )
    if args.run and kind is AttemptKind.SCORE:
        score_item(console, args, model, prompt, item, items)
    if args.run and kind is AttemptKind.GENERATE:
        generate_item(console, args, model, prompt, item, items)


if __name__ == "__main__":
    main()
