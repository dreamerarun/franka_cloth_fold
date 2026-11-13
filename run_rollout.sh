#!/usr/bin/env bash
# usage: bash run_rollout.sh <tries> <seed> <logfile> [extra args for 02_collect_franka.py]
# env overrides: ISAAC=... CONDA_ENV=... CKPT=... PORT=...
TRIES=$1; SEED=$2; LOG=$3; shift 3
WAM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC=${ISAAC:-$HOME/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64}
CONDA_ENV=${CONDA_ENV:-lerobot}
CKPT=${CKPT:-$WAM/outputs/train/wam_fold_act/checkpoints/last/pretrained_model}
PORT=${PORT:-6000}
CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
( source "$CONDA_SH"; conda activate "$CONDA_ENV"; cd "$WAM"
  exec python policy_server.py "$CKPT" --port "$PORT" --device cpu > /tmp/policy_server.log 2>&1 ) &
SRV=$!
for i in $(seq 1 60); do grep -q "waiting on port" /tmp/policy_server.log 2>/dev/null && break; sleep 2; done
grep -q "waiting on port" /tmp/policy_server.log || { echo "server did not start:"; tail -n 20 /tmp/policy_server.log; kill $SRV; exit 1; }
cd "$ISAAC"
PYTHONUNBUFFERED=1 ./python.sh "$WAM/02_collect_franka.py" --headless --policy-port "$PORT" --episodes 1000 --max-tries "$TRIES" --seed "$SEED" --out /tmp/rollout_out_$SEED "$@" > "$LOG" 2>&1
kill $SRV 2>/dev/null
grep -a "^\[done\]" "$LOG"
