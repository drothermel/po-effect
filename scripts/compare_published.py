# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datadec",
#     "pyarrow>=26.0.0",
# ]
#
# [tool.uv.sources]
# datadec = { git = "https://github.com/danielle-rothermel/datadec" }
# ///
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from datadec.data.artifacts import DataArtifacts
from datadec.data.download import download_processed_outputs
from datadec.data.selection import resolve_selection
from datadec.eval.scoring import ChoiceScore
from datadec.eval.verify_converted import (
    PublishedKey,
    ScoredItem,
    compare,
    load_published,
)
from datadec.models import DataDecideCheckpoint

DATADEC_DATA_DIR = Path("~/.cache/po-effect/datadec").expanduser()


def scored_items(path: Path) -> tuple[ScoredItem, ...]:
    rows = pq.read_table(path).to_pylist()
    return tuple(
        ScoredItem(
            native_id=row["native_id"],
            gold_index=row["gold_index"],
            choices=tuple(
                ChoiceScore(
                    log_likelihood=log_likelihood,
                    token_count=token_count,
                    char_count=char_count,
                    byte_count=byte_count,
                    unconditional_log_likelihood=unconditional,
                )
                for (
                    log_likelihood,
                    token_count,
                    char_count,
                    byte_count,
                    unconditional,
                ) in zip(
                    row["log_likelihood"],
                    row["token_count"],
                    row["char_count"],
                    row["byte_count"],
                    row["unconditional_log_likelihood"],
                    strict=True,
                )
            ),
            conditional_input_tokens=0,
            warnings=0,
        )
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a po-effect ARC-Challenge RC parquet export against "
            "published DataDecide instance values."
        )
    )
    parser.add_argument("export", type=Path)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATADEC_DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    artifacts = DataArtifacts(data_dir)
    print(
        f"ensuring published {args.recipe} OLMES detail tables in {data_dir}"
    )
    download_processed_outputs(
        artifacts, resolve_selection(olmes_details=(args.recipe,))
    )

    checkpoint = DataDecideCheckpoint(
        recipe=args.recipe,
        size=args.size,
        seed_ordinal=args.seed,
        step=args.step,
    )
    key = PublishedKey(
        recipe=checkpoint.recipe,
        params=checkpoint.size,
        seed=checkpoint.seed.name,
        step=checkpoint.step,
    )
    ours = scored_items(args.export)
    kept = {item.native_id for item in ours}
    published = {
        native_id: item
        for native_id, item in load_published(artifacts, key).items()
        if native_id in kept
    }
    comparison = compare(ours, published)

    print(f"items {comparison.items}, choices {comparison.choices}")
    print(f"{'rule':<10} {'agree':>8} {'ours':>8} {'published':>10}")
    for rule in comparison.rules:
        print(
            f"{rule.rule.value:<10} {rule.agreement:>8.4f} "
            f"{rule.accuracy_ours:>8.4f} {rule.accuracy_published:>10.4f}"
        )
        for mismatch in rule.first_mismatches:
            print(f"    {mismatch}")
    print(
        "sum_logits abs diff mean "
        f"{comparison.sum_logits_mean_abs_diff:.6f} "
        f"max {comparison.sum_logits_max_abs_diff:.6f}"
    )
    print(
        "sum_logits_uncond abs diff mean "
        f"{comparison.sum_logits_uncond_mean_abs_diff:.6f} "
        f"max {comparison.sum_logits_uncond_max_abs_diff:.6f}"
    )
    print(
        "continuation token agreement "
        f"{comparison.continuation_token_agreement:.4f}"
    )
    for name, passed in comparison.acceptance.items():
        print(f"{name} {'pass' if passed else 'fail'}")


if __name__ == "__main__":
    main()
