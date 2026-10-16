#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: $0 QUEUE" >&2
    exit 2
fi
QUEUE=$1
POOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$POOL_DIR/env.sh"
squeue -u "$USER" -n "$POOL_NAME" -o "%.12i %.12P %.20j %.8T %.12M %.12l %R"
for state in pending running done failed; do
    count=$(find "$QUEUE/$state" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    printf '%-8s %d\n' "$state" "$count"
done
