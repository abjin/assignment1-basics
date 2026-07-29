#!/bin/bash
# Sequential experiment runner: waits for the current training process to
# finish, then runs ablations, LR sweep, batch-size runs, and OWT training.
set -u
TS_DATA=/notebooks/data/tinystories
OWT_DATA=/notebooks/data/owt
RUNS=/notebooks/runs

echo "WAITING_FOR_BASE $(date +%H:%M)"
while pgrep -f "cs336_basics.train" > /dev/null; do sleep 60; done
echo "BASE_FINISHED $(date +%H:%M)"

run() {
  local name=$1; shift
  if [ -f "$RUNS/$name/final.pt" ]; then echo "SKIP $name (already done)"; return; fi
  local resume=()
  if [ -f "$RUNS/$name/latest.pt" ]; then resume=(--resume "$RUNS/$name/latest.pt"); echo "RESUME $name"; fi
  echo "START $name $(date +%H:%M)"
  python3 -m cs336_basics.train --data-dir "$TS_DATA" --out-dir "$RUNS/$name" "${resume[@]}" "$@" \
    >> "$RUNS/$name.log" 2>&1
  local last_val
  last_val=$(grep val_loss "$RUNS/$name/log.jsonl" 2>/dev/null | tail -1)
  echo "DONE $name $(date +%H:%M) $last_val"
}

# Ablations (handout section 7.3)
run ablation-no-rmsnorm --no-rmsnorm
run ablation-no-rmsnorm-lowlr --no-rmsnorm --lr 1e-4
run ablation-post-norm --post-norm
run ablation-nope --no-rope
run ablation-silu --ffn-type silu --d-ff 2048

# Learning rate sweep (base run at 1e-3 already exists)
run lr-3e-4 --lr 3e-4
run lr-3e-3 --lr 3e-3
run lr-1e-2-divergent --lr 1e-2 --total-tokens 32768000

# Batch size variations (total tokens fixed; LR scaled with batch size)
run batch-32 --batch-size 32 --lr 5e-4
run batch-256 --batch-size 256 --lr 2e-3

# OWT main experiment (wait for encoded data)
echo "WAITING_FOR_OWT_DATA $(date +%H:%M)"
while [ ! -f "$OWT_DATA/valid.npy" ]; do sleep 120; done
if [ ! -f "$RUNS/owt-base/final.pt" ]; then
  owt_resume=()
  if [ -f "$RUNS/owt-base/latest.pt" ]; then owt_resume=(--resume "$RUNS/owt-base/latest.pt"); echo "RESUME owt-base"; fi
  echo "START owt-base $(date +%H:%M)"
  python3 -m cs336_basics.train --data-dir "$OWT_DATA" --out-dir "$RUNS/owt-base" \
    --vocab-size 32000 "${owt_resume[@]}" >> "$RUNS/owt-base.log" 2>&1
  echo "DONE owt-base $(date +%H:%M) $(grep val_loss "$RUNS/owt-base/log.jsonl" | tail -1)"
fi

echo "ALL_EXPERIMENTS_DONE $(date +%H:%M)"
