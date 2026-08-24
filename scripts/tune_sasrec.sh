#!/usr/bin/env bash
# Small validation grid for SASRec (dimension x dropout) on the leave-last-out split.
# The test set is not evaluated here. The best setting per category goes into run_matrix.sh.
#
#   scripts/tune_sasrec.sh <gpu> <data>
set -u
GPU=${1:?gpu index}; DATA=${2:?dataset name}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs
for dim in 64 128; do
  for drop in 0.1 0.2 0.5; do
    name="tune/${DATA}_sasrec_d${dim}_p${drop}"
    uv run python -m mmsid.train_seq sasrec name="$name" data="$DATA" split=last_out \
      model.sasrec.dim=$dim model.sasrec.dropout=$drop eval.test=false \
      > "logs/tune_${DATA}_d${dim}_p${drop}.log" 2>&1
  done
done
uv run python - "$DATA" <<'PY'
import json, sys, glob
rows = []
for path in glob.glob(f"results/tune/{sys.argv[1]}_sasrec_*/seed0/metrics.json"):
    m = json.load(open(path))
    rows.append((m["valid"]["ndcg@10"], m["valid"]["recall@10"], m["name"]))
for ndcg, recall, name in sorted(rows, reverse=True):
    print(f"{name:45s} valid NDCG@10 {ndcg:.4f}  Recall@10 {recall:.4f}")
PY
