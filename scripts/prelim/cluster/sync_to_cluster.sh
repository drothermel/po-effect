#!/bin/bash
set -euo pipefail
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [JOB_ID]" >&2
    exit 2
fi
JOB_ID=${1:-}
HOST=torch
REMOTE=/scratch/ddr8143/drotherm/repos/po-effect
SOURCE=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
SYNC="uv sync --locked --python 3.13"
ssh "$HOST" "mkdir -p $REMOTE"
git -C "$SOURCE" ls-files -z \
    | rsync -az --from0 --files-from=- --ignore-missing-args "$SOURCE/" "$HOST:$REMOTE/"
echo "synced tracked files from $SOURCE to $HOST:$REMOTE"
ENV_SETUP="source $REMOTE/scripts/prelim/cluster/env.sh"
if [[ -n "$JOB_ID" ]]; then
    ssh "$HOST" "$ENV_SETUP && cd $REMOTE && srun --jobid=$JOB_ID --overlap --ntasks=1 $SYNC"
else
    echo "no job id given; inside an allocation run:"
    echo "  $ENV_SETUP && cd $REMOTE && $SYNC"
fi
