#!/bin/bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
    echo "usage: $0 ACCOUNT PARTITION N QUEUE LOGS" >&2
    exit 2
fi
ACCOUNT=$1
PARTITION=$2
N=$3
QUEUE=$4
LOGS=$5
POOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$POOL_DIR/env.sh"
CILVR_PARTITION=a100_cilvr
CILVR_MAX_POOLS=4
if [[ ! "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "N must be a positive integer, got $N" >&2
    exit 2
fi
if [[ "$PARTITION" == "$CILVR_PARTITION" ]]; then
    existing=$(squeue -h -u "$USER" -p "$PARTITION" -n "$POOL_NAME" | wc -l)
    if (( existing + N > CILVR_MAX_POOLS )); then
        echo "refusing: $existing pools on $PARTITION plus $N exceeds $CILVR_MAX_POOLS" >&2
        exit 1
    fi
fi
mkdir -p "$LOGS"
for _ in $(seq "$N"); do
    sbatch --parsable \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --job-name "$POOL_NAME" \
        --output "$LOGS/$POOL_NAME-%j.out" \
        --export "ALL,QUEUE=$QUEUE,LOGS=$LOGS,POOL_DIR=$POOL_DIR,POOL_NAME=$POOL_NAME" \
        "$POOL_DIR/pool.sbatch"
done
