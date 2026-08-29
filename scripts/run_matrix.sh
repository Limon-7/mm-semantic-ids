#!/usr/bin/env bash
# Run the experiment matrix: four ID variants and the baselines, on two categories and two splits.
#
#   scripts/run_matrix.sh <gpu> [worker] [n_workers]
#
# Jobs are dealt out round-robin, so several workers can share the list:
#   scripts/run_matrix.sh 0 0 3 &  scripts/run_matrix.sh 0 1 3 &  scripts/run_matrix.sh 1 2 3 &
# A run whose results/<name>/seed<k>/metrics.json exists is skipped, so the script can be restarted.
# ONLY is an optional regex on the run name, e.g. ONLY='_(T|I|EF|LF)$' for the generative runs.
set -u
GPU=${1:?gpu index}; WORKER=${2:-0}; N_WORKERS=${3:-1}; SEED=${SEED:-0}; ONLY=${ONLY:-.}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

declare -A SHORT=([video_games]=vg [toys_and_games]=toys)
declare -A SPLIT=([last_out]=lastout [timestamp]=time)
# tuned on validation, see scripts/tune_sasrec.sh and docs/notes.md
declare -A SASREC=(
  [video_games]="${SASREC_VG:-model.sasrec.dim=64 model.sasrec.dropout=0.2}"
  [toys_and_games]="${SASREC_TOYS:-model.sasrec.dim=64 model.sasrec.dropout=0.1}"
)

jobs=()
for data in video_games toys_and_games; do
  for split in last_out timestamp; do
    tag="${SHORT[$data]}_${SPLIT[$split]}"
    jobs+=("popularity name=${tag}_pop data=$data split=$split")
    for variant in T I EF LF; do
      jobs+=("tiger name=${tag}_${variant} data=$data split=$split ids.spec.variant=$variant")
    done
    for input in id text text_image id_text_image; do
      jobs+=("sasrec name=${tag}_sasrec_${input} data=$data split=$split model.sasrec.item_input=$input ${SASREC[$data]}")
    done
  done
done

selected=()
for job in "${jobs[@]}"; do
  read -r -a args <<< "$job"
  [[ ${args[1]#name=} =~ $ONLY ]] && selected+=("$job")
done

for i in "${!selected[@]}"; do
  (( i % N_WORKERS == WORKER )) || continue
  read -r -a args <<< "${selected[$i]}"
  name=${args[1]#name=}
  echo "[$(date +%H:%M)] gpu $GPU: $name"
  uv run python -m mmsid.train_seq "${args[@]}" seed="$SEED" > "logs/${name}_seed${SEED}.log" 2>&1 \
    || echo "  FAILED, see logs/${name}_seed${SEED}.log"
done
