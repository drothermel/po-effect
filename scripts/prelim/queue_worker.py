from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from queue_layout import JOB_SUFFIX, QueueState, job_file

DEFAULT_IDLE_EXIT_S = 900.0
DEFAULT_POLL_S = 10.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Claim jobs from <queue>/pending/ by atomic rename into "
            "running/, run each command in a subprocess with its output in "
            "<log-dir>/<name>.log, move the job record with its exit code "
            "into done/ or failed/, and exit after --idle-exit-s seconds "
            "with nothing claimable. One job at a time."
        )
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--idle-exit-s", type=float, default=DEFAULT_IDLE_EXIT_S
    )
    parser.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="claim jobs marked needs_gpu as well as the rest",
    )
    return parser.parse_args(argv)


def claim(queue: Path, *, gpu: bool) -> tuple[dict[str, Any], Path] | None:
    pending = queue / QueueState.PENDING.value
    for path in sorted(pending.glob(f"*{JOB_SUFFIX}")):
        try:
            job = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        if job["needs_gpu"] and not gpu:
            continue
        target = job_file(queue, QueueState.RUNNING, job["name"])
        try:
            path.rename(target)
        except FileNotFoundError:
            continue
        return job, target
    return None


def run(job: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    log_path = log_dir / f"{job['name']}.log"
    started = time.time()
    with log_path.open("w") as log:
        completed = subprocess.run(  # noqa: S603
            job["command"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        **job,
        "worker": f"{socket.gethostname()}:{os.getpid()}",
        "started": started,
        "finished": time.time(),
        "exit_code": completed.returncode,
        "log": str(log_path),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for state in QueueState:
        (args.queue / state.value).mkdir(parents=True, exist_ok=True)
    idle_since = time.monotonic()
    while True:
        claimed = claim(args.queue, gpu=args.gpu)
        if claimed is None:
            if time.monotonic() - idle_since >= args.idle_exit_s:
                print("idle timeout, exiting", flush=True)
                return
            time.sleep(args.poll_s)
            continue
        job, running = claimed
        print(f"running {job['name']}", flush=True)
        try:
            record = run(job, args.log_dir)
        except OSError as error:
            record = {**job, "exit_code": None, "error": repr(error)}
        state = (
            QueueState.DONE if record["exit_code"] == 0 else QueueState.FAILED
        )
        target = job_file(args.queue, state, job["name"])
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(target)
        running.unlink()
        print(
            f"{state.value} {job['name']} ({record['exit_code']})", flush=True
        )
        idle_since = time.monotonic()


if __name__ == "__main__":
    main()
