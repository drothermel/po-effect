#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -euo pipefail
: "${QUEUE:?QUEUE must be set}"
: "${LOGS:?LOGS must be set}"
: "${POOL_DIR:?POOL_DIR must be set}"
source "$POOL_DIR/env.sh"
if [[ ! -f "$SECRETS" ]]; then
    echo "missing $SECRETS; it must contain: export OPENROUTER_API_KEY=..." >&2
    exit 1
fi
source "$SECRETS"
: "${OPENROUTER_API_KEY:?$SECRETS must export OPENROUTER_API_KEY}"
export UV_NO_SYNC=1
cd "$CLONE"
exec uv run python scripts/prelim/queue_worker.py --queue "$QUEUE" --log-dir "$LOGS" --idle-exit-s 900 --gpu
