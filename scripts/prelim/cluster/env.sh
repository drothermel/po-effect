export SCRATCH=/scratch/ddr8143
restore_nounset=$(set +o | grep nounset)
set +u
source "$SCRATCH/.config/shell/cluster_env.sh"
eval "$restore_nounset"
export PATH=/home/ddr8143/.local/bin:$PATH
export HOME=/scratch/ddr8143
CLONE=$SCRATCH/drotherm/repos/po-effect
SECRETS=$SCRATCH/.config/secrets/openrouter.env
POOL_NAME=${POOL_NAME:-po-effect-pool}
