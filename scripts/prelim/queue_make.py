from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from queue_layout import STATES, QueueState, job_file

JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class JobEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=JOB_NAME.pattern)
    command: tuple[str, ...] = Field(min_length=1)
    needs_gpu: bool = False
    model: str | None = None


class JobsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    jobs: tuple[JobEntry, ...] = Field(min_length=1)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a jobs TOML file (a list of named commands with optional "
            "needs_gpu and model) and write one JSON file per job into "
            "<queue>/pending/, substituting {out_root} and {name} in each "
            "command argument; jobs already present in any queue state are "
            "skipped."
        )
    )
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--out-root",
        required=True,
        help="value substituted for {out_root} in commands",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with args.jobs.open("rb") as stream:
        jobs = JobsFile.model_validate(tomllib.load(stream)).jobs
    names = [job.name for job in jobs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"duplicate job names: {duplicates}")
    for state in STATES:
        (args.queue / state.value).mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for job in jobs:
        if any(
            job_file(args.queue, state, job.name).exists() for state in STATES
        ):
            skipped += 1
            continue
        command = [
            part.format(out_root=args.out_root, name=job.name)
            for part in job.command
        ]
        payload = {
            "name": job.name,
            "command": command,
            "needs_gpu": job.needs_gpu,
            "model": job.model,
        }
        target = job_file(args.queue, QueueState.PENDING, job.name)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(target)
        written += 1
    print(f"wrote {written} jobs, skipped {skipped} already queued")


if __name__ == "__main__":
    main()
