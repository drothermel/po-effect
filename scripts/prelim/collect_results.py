from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, fields
from enum import UNIQUE, StrEnum, auto, verify
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

RESULTS_FILE = "results.json"
PROGRESS_FILE = "progress.log"
ARGS_FILE = "args.json"
ARC_RESPONSES_FILE = "responses.jsonl"
ARC_TASK = "arc"
HELDOUT = "heldout"
MISSING = "-"
THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
ANSWER_PREFIX = re.compile(
    r"^[\s*_#>(\[]*(?:(?:the\s+)?(?:correct\s+|final\s+)?answer"
    r"(?:\s+is)?|option|choice)\s*[:\-]?\s*",
    re.IGNORECASE,
)
ANSWER_LETTER = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")
OPENROUTER_START = re.compile(
    r"start: arm (?P<arm>\S+), task (?P<model>\S+), reflection \S+, "
    r"prompt (?P<prompt>\S+), splits (?P<splits>\{.*\})"
)
LOCAL_START = re.compile(
    r"start: arm (?P<arm>\S+), model (?P<model>\S+), (?P<mode>.*), "
    r"reflection \S+, prompt (?P<prompt>\S+), splits (?P<splits>\{.*\})"
)
ARC_SCREEN = re.compile(r"screen: seed val (?P<val>[\d.]+), in band (\w+)")
ARC_ACCEPTED = re.compile(r"gepa accepted candidate|null candidate \d+")
METRIC_CALLS = re.compile(r"metric calls (\d+)")
LEGACY_RULE = "rule "


@verify(UNIQUE)
class Source(StrEnum):
    ARC_OPENROUTER = auto()
    ARC_LOCAL = auto()
    FAMILY = auto()


@verify(UNIQUE)
class RunStatus(StrEnum):
    COMPLETE = auto()
    OUT_OF_BAND = auto()
    SEARCH_FAILED = auto()
    IN_PROGRESS = auto()


@verify(UNIQUE)
class Attempt(StrEnum):
    SCORE = auto()
    GENERATE = auto()


RAW_STATUS = {
    "complete": RunStatus.COMPLETE,
    "done": RunStatus.COMPLETE,
    "out_of_band": RunStatus.OUT_OF_BAND,
    "search_failed": RunStatus.SEARCH_FAILED,
    "running": RunStatus.IN_PROGRESS,
}


@dataclass(frozen=True, slots=True)
class RunRow:
    source: str
    task: str
    task_model: str | None
    reasoning: str | None
    attempt: str | None
    rule: str | None
    prompt: str | None
    arm: str | None
    seed: int | None
    budget: int | None
    search_calls: int | None
    screen: float | None
    in_band: bool | None
    seed_heldout: float | None
    best_heldout: float | None
    delta: float | None
    heldout_n: int | None
    candidates_accepted: int | None
    wall_s: float | None
    cost_usd: float | None
    status: str
    run: str
    run_dir: str


SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("task", pa.string()),
        ("task_model", pa.string()),
        ("reasoning", pa.string()),
        ("attempt", pa.string()),
        ("rule", pa.string()),
        ("prompt", pa.string()),
        ("arm", pa.string()),
        ("seed", pa.int64()),
        ("budget", pa.int64()),
        ("search_calls", pa.int64()),
        ("screen", pa.float64()),
        ("in_band", pa.bool_()),
        ("seed_heldout", pa.float64()),
        ("best_heldout", pa.float64()),
        ("delta", pa.float64()),
        ("heldout_n", pa.int64()),
        ("candidates_accepted", pa.int64()),
        ("wall_s", pa.float64()),
        ("cost_usd", pa.float64()),
        ("status", pa.string()),
        ("run", pa.string()),
        ("run_dir", pa.string()),
    ]
)
COLUMNS = tuple(field.name for field in fields(RunRow))


@dataclass(frozen=True, slots=True)
class FormatShare:
    run: RunRow
    n: int
    seed_strict: float
    seed_lenient: float
    best_strict: float
    best_lenient: float


class UnparsedRunError(ValueError):
    pass


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect prelim sweep results from gepa_arc_openrouter.py, "
            "gepa_arc_local.py, and whetstone-envs gepa_family.py run "
            "directories under the roots into runs.csv, runs.parquet, "
            "summary.md, and format_share.md under --out."
        )
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def run_dirs(roots: Iterable[Path], out: Path) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        for name in (RESULTS_FILE, PROGRESS_FILE):
            found.update(path.parent.resolve() for path in root.rglob(name))
    excluded = out.resolve()
    return sorted(
        path
        for path in found
        if path != excluded and excluded not in path.parents
    )


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UnparsedRunError(f"{path.name} is not a JSON object")
    return payload


def log_lines(run_dir: Path) -> list[str]:
    path = run_dir / PROGRESS_FILE
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def status_of(raw: object, best_heldout: object) -> RunStatus:
    if not isinstance(raw, str) or raw not in RAW_STATUS:
        raise UnparsedRunError(f"unknown status {raw!r}")
    status = RAW_STATUS[raw]
    if best_heldout is None and status is not RunStatus.OUT_OF_BAND:
        return RunStatus.IN_PROGRESS
    return status


def optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def api_cost(api: object) -> float | None:
    if not isinstance(api, dict):
        return None
    costs = [
        float(stats["cost_usd"])
        for stats in api.values()
        if isinstance(stats, dict) and "cost_usd" in stats
    ]
    return sum(costs) if costs else None


def accepted(candidates: object) -> int | None:
    if isinstance(candidates, list):
        return max(len(candidates) - 1, 0)
    if isinstance(candidates, int):
        return max(candidates - 1, 0)
    return None


def arc_source(results: Mapping[str, Any]) -> Source:
    if "task_model" in results:
        return Source.ARC_OPENROUTER
    if "model" in results:
        return Source.ARC_LOCAL
    raise UnparsedRunError("results.json names neither task_model nor model")


def arc_attempt(results: Mapping[str, Any], source: Source) -> Attempt:
    if source is Source.ARC_OPENROUTER:
        return Attempt.GENERATE
    raw = results.get("attempt")
    if raw is None and "rule" in results:
        return Attempt.SCORE
    return Attempt(raw)


def arc_row(run_dir: Path, results: Mapping[str, Any]) -> RunRow:
    source = arc_source(results)
    screen = results.get("screen") or {}
    best = results.get("best_heldout")
    return RunRow(
        source=source.value,
        task=ARC_TASK,
        task_model=results.get("task_model") or results.get("model"),
        reasoning=results.get("reasoning"),
        attempt=arc_attempt(results, source).value,
        rule=results.get("rule") or results.get("readout"),
        prompt=results.get("prompt"),
        arm=results.get("arm"),
        seed=optional_int(results.get("seed")),
        budget=optional_int(results.get("max_metric_calls")),
        search_calls=optional_int(results.get("search_metric_calls")),
        screen=optional_float(screen.get("seed_val")),
        in_band=screen.get("in_band"),
        seed_heldout=optional_float(results.get("seed_heldout")),
        best_heldout=optional_float(best),
        delta=optional_float(results.get("delta")),
        heldout_n=optional_int((results.get("splits") or {}).get(HELDOUT)),
        candidates_accepted=accepted(results.get("candidates")),
        wall_s=optional_float(results.get("wall_s")),
        cost_usd=api_cost(results.get("api")),
        status=status_of(results.get("status"), best).value,
        run=run_dir.name,
        run_dir=str(run_dir),
    )


def local_mode(mode: str) -> tuple[Attempt, str | None]:
    if mode.startswith(LEGACY_RULE):
        return Attempt.SCORE, mode.removeprefix(LEGACY_RULE)
    described = ast.literal_eval(mode)
    if not isinstance(described, dict):
        raise UnparsedRunError(f"unreadable mode {mode!r}")
    return (
        Attempt(described["attempt"]),
        described.get("rule") or described.get("readout"),
    )


def arc_progress_row(run_dir: Path, lines: Sequence[str]) -> RunRow:
    head = lines[0] if lines else ""
    if match := OPENROUTER_START.search(head):
        source, attempt, rule = Source.ARC_OPENROUTER, Attempt.GENERATE, None
    elif match := LOCAL_START.search(head):
        source = Source.ARC_LOCAL
        attempt, rule = local_mode(match["mode"])
    else:
        raise UnparsedRunError("no results.json and no recognized start line")
    screen = next(
        (found for line in lines if (found := ARC_SCREEN.search(line))), None
    )
    calls = [int(m[1]) for line in lines if (m := METRIC_CALLS.search(line))]
    splits = ast.literal_eval(match["splits"])
    return RunRow(
        source=source.value,
        task=ARC_TASK,
        task_model=match["model"],
        reasoning=None,
        attempt=attempt.value,
        rule=rule,
        prompt=match["prompt"],
        arm=match["arm"],
        seed=None,
        budget=None,
        search_calls=calls[-1] if calls else None,
        screen=None if screen is None else float(screen["val"]),
        in_band=None if screen is None else screen[2] == str(True),
        seed_heldout=None,
        best_heldout=None,
        delta=None,
        heldout_n=optional_int(splits.get(HELDOUT)),
        candidates_accepted=sum(bool(ARC_ACCEPTED.search(x)) for x in lines),
        wall_s=None,
        cost_usd=None,
        status=RunStatus.IN_PROGRESS.value,
        run=run_dir.name,
        run_dir=str(run_dir),
    )


def family_cost(results: Mapping[str, Any]) -> float | None:
    parts = [
        results.get(key) for key in ("task_cost_usd", "reflection_cost_usd")
    ]
    costs = [float(part) for part in parts if part is not None]
    return sum(costs) if costs else None


def family_row(
    run_dir: Path,
    results: Mapping[str, Any] | None,
    args: Mapping[str, Any] | None,
) -> RunRow:
    merged = {**(args or {}), **(results or {})}
    if "family" not in merged:
        raise UnparsedRunError("family run names no family")
    best = merged.get("best_heldout")
    sizes = merged.get("sizes") or {}
    raw_status = merged.get("status", "running")
    return RunRow(
        source=Source.FAMILY.value,
        task=merged["family"],
        task_model=merged.get("task_model"),
        reasoning=merged.get("reasoning"),
        attempt=Attempt.GENERATE.value,
        rule=None,
        prompt=None,
        arm=merged.get("arm"),
        seed=optional_int(merged.get("seed")),
        budget=optional_int(merged.get("max_metric_calls")),
        search_calls=optional_int(merged.get("metric_calls")),
        screen=optional_float(merged.get("naive_val")),
        in_band=merged.get("in_band"),
        seed_heldout=optional_float(merged.get("naive_heldout")),
        best_heldout=optional_float(best),
        delta=optional_float(merged.get("delta_heldout")),
        heldout_n=optional_int(sizes.get(HELDOUT, merged.get("heldout_n"))),
        candidates_accepted=accepted(merged.get("candidates")),
        wall_s=optional_float(merged.get("wall_s")),
        cost_usd=family_cost(merged),
        status=status_of(raw_status, best).value,
        run=run_dir.name,
        run_dir=str(run_dir),
    )


def collect_run(run_dir: Path) -> RunRow:
    results = read_json(run_dir / RESULTS_FILE)
    args = read_json(run_dir / ARGS_FILE)
    if args is not None or (results is not None and "family" in results):
        return family_row(run_dir, results, args)
    if results is not None:
        return arc_row(run_dir, results)
    return arc_progress_row(run_dir, log_lines(run_dir))


def lenient_letter(text: str) -> str | None:
    stripped = THINK_BLOCK.sub("", text).strip()
    if not stripped:
        return None
    first = ANSWER_PREFIX.sub("", stripped.splitlines()[0].strip())
    match = ANSWER_LETTER.search(first)
    return None if match is None else match[1]


def response_text(response: object) -> str | None:
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]
    return None


def heldout_scores(
    rows: Sequence[Mapping[str, Any]], identifier: str
) -> tuple[list[float], list[float]] | None:
    chosen = {
        row["item_id"]: row
        for row in rows
        if row.get("instruction_id") == identifier
        and row.get("split") == HELDOUT
    }
    strict: list[float] = []
    lenient: list[float] = []
    for row in chosen.values():
        text = response_text(row.get("response"))
        if text is None:
            return None
        strict.append(float(row["score"]))
        lenient.append(float(lenient_letter(text) == row["gold"]))
    return (strict, lenient) if strict else None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def format_share(run: RunRow) -> FormatShare | None:
    run_dir = Path(run.run_dir)
    results = read_json(run_dir / RESULTS_FILE) or {}
    seed_id = results.get("seed_instruction_id")
    best_id = results.get("best_instruction_id")
    if not isinstance(seed_id, str) or not isinstance(best_id, str):
        return None
    rows = list(jsonl(run_dir / ARC_RESPONSES_FILE))
    seed = heldout_scores(rows, seed_id)
    best = heldout_scores(rows, best_id)
    if seed is None or best is None:
        return None
    return FormatShare(
        run=run,
        n=len(seed[0]),
        seed_strict=mean(seed[0]),
        seed_lenient=mean(seed[1]),
        best_strict=mean(best[0]),
        best_lenient=mean(best[1]),
    )


def sort_key(run: RunRow) -> tuple[str, str, int, int, str]:
    return (
        run.task_model or "",
        run.arm or "",
        run.budget if run.budget is not None else sys.maxsize,
        run.seed if run.seed is not None else sys.maxsize,
        run.run,
    )


def cell(value: object) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def signed(value: float) -> str:
    return f"{value:+.3f}"


def table(header: Sequence[str], body: Iterable[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def grouped(runs: Iterable[RunRow]) -> dict[tuple[str, str], list[RunRow]]:
    groups: dict[tuple[str, str], list[RunRow]] = {}
    for run in runs:
        groups.setdefault((run.source, run.task), []).append(run)
    return {key: sorted(groups[key], key=sort_key) for key in sorted(groups)}


SUMMARY_COLUMNS = (
    "task_model",
    "reasoning",
    "attempt",
    "rule",
    "prompt",
    "arm",
    "seed",
    "budget",
    "search_calls",
    "screen",
    "in_band",
    "seed_heldout",
    "best_heldout",
    "delta",
    "heldout_n",
    "candidates_accepted",
    "wall_s",
    "cost_usd",
    "status",
    "run",
)


def summary_lines(
    runs: Sequence[RunRow], problems: Sequence[tuple[Path, str]]
) -> list[str]:
    lines = ["# Prelim runs", ""]
    if not runs:
        lines.extend(["No runs found.", ""])
    for (source, task), members in grouped(runs).items():
        lines.extend([f"## {source} / {task}", ""])
        lines.extend(
            table(
                SUMMARY_COLUMNS,
                (
                    [cell(getattr(run, name)) for name in SUMMARY_COLUMNS]
                    for run in members
                ),
            )
        )
        lines.append("")
    if problems:
        lines.extend(["## Unparsed run directories", ""])
        lines.extend(f"- `{path}`: {reason}" for path, reason in problems)
        lines.append("")
    return lines


SHARE_COLUMNS = (
    "task_model",
    "attempt",
    "prompt",
    "arm",
    "seed",
    "budget",
    "n",
    "seed strict",
    "seed lenient",
    "best strict",
    "best lenient",
    "delta strict",
    "delta lenient",
    "format part",
    "run",
)


def share_row(share: FormatShare) -> list[str]:
    run = share.run
    strict = share.best_strict - share.seed_strict
    lenient = share.best_lenient - share.seed_lenient
    return [
        cell(run.task_model),
        cell(run.attempt),
        cell(run.prompt),
        cell(run.arm),
        cell(run.seed),
        cell(run.budget),
        str(share.n),
        cell(share.seed_strict),
        cell(share.seed_lenient),
        cell(share.best_strict),
        cell(share.best_lenient),
        signed(strict),
        signed(lenient),
        signed(strict - lenient),
        run.run,
    ]


def format_share_lines(
    shares: Sequence[FormatShare], skipped: Sequence[RunRow]
) -> list[str]:
    lines = [
        "# Format share of held-out deltas",
        "",
        (
            "Strict is the runner's own score. Lenient reads the first "
            "letter A-E in the first line after removing think blocks and "
            "answer prefixes. Format part is delta strict minus delta "
            "lenient."
        ),
        "",
    ]
    if not shares:
        lines.extend(["No runs with seed and best responses.", ""])
    by_run = {share.run.run_dir: share for share in shares}
    for (source, task), members in grouped(s.run for s in shares).items():
        lines.extend([f"## {source} / {task}", ""])
        lines.extend(
            table(
                SHARE_COLUMNS,
                (share_row(by_run[run.run_dir]) for run in members),
            )
        )
        lines.append("")
    if skipped:
        lines.extend(["## Finished runs without a lenient rescore", ""])
        for (source, task), members in grouped(skipped).items():
            reason = (
                "no lenient rule saved by the runner"
                if source == Source.FAMILY.value
                else "no per-item seed and best text responses"
            )
            lines.append(f"- {source} / {task}: {len(members)} ({reason})")
        lines.append("")
    return lines


def rescorable(run: RunRow) -> bool:
    return (
        run.source != Source.FAMILY.value
        and run.attempt == Attempt.GENERATE.value
        and run.best_heldout is not None
    )


def write_tables(runs: Sequence[RunRow], out: Path) -> None:
    records = [asdict(run) for run in runs]
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    pq.write_table(
        pa.Table.from_pylist(records, schema=SCHEMA), out / "runs.parquet"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    runs: list[RunRow] = []
    problems: list[tuple[Path, str]] = []
    for run_dir in run_dirs(args.roots, args.out):
        try:
            runs.append(collect_run(run_dir))
        except (UnparsedRunError, ValueError, KeyError, TypeError) as error:
            problems.append((run_dir, f"{type(error).__name__}: {error}"))
    runs.sort(key=lambda run: (run.source, run.task, *sort_key(run)))
    shares: list[FormatShare] = []
    skipped: list[RunRow] = []
    for run in runs:
        share = format_share(run) if rescorable(run) else None
        if share is not None:
            shares.append(share)
        elif run.best_heldout is not None:
            skipped.append(run)
    write_tables(runs, args.out)
    (args.out / "summary.md").write_text(
        "\n".join(summary_lines(runs, problems)), encoding="utf-8"
    )
    (args.out / "format_share.md").write_text(
        "\n".join(format_share_lines(shares, skipped)), encoding="utf-8"
    )
    counts = {
        source.value: sum(run.source == source.value for run in runs)
        for source in Source
    }
    print(
        f"runs {counts}; format share {len(shares)}; unparsed {len(problems)}"
    )
    for path, reason in problems:
        print(f"unparsed {path}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
