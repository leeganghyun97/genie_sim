#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_cli_present=false

for argument in "$@"; do
  case "${argument}" in
    --resume|--resume=*|--checkpoint|--checkpoint=*|--append)
      echo "preloss_v2 is a fresh-training recipe; checkpoint/replay restore is forbidden" >&2
      exit 2
      ;;
    --demonstration-dataset|--demonstration-dataset=*)
      dataset_cli_present=true
      ;;
    --recurrent-profile|--recurrent-profile=*|--batch-size|--batch-size=*|\
    --camera-encoder-profile|--camera-encoder-profile=*|\
    --random-shift-pad|--random-shift-pad=*|\
    --relative-pose-weight|--relative-pose-weight=*|\
    --pose-consistency-weight|--pose-consistency-weight=*|\
    --pose-consistency-rotation-weight|--pose-consistency-rotation-weight=*|\
    --contact-weight|--contact-weight=*|\
    --depth-validity-weight|--depth-validity-weight=*|\
    --student-head-distillation-weight|--student-head-distillation-weight=*|\
    --cross-camera-contrastive-weight|--cross-camera-contrastive-weight=*|\
    --temporal-pose-residual-weight|--temporal-pose-residual-weight=*)
      echo "${argument} is sealed by the preloss_v2 training recipe" >&2
      exit 2
      ;;
  esac
done

dataset_arguments=()
if [[ -n "${G2_DEMONSTRATION_DATASETS:-}" ]]; then
  IFS=':' read -r -a dataset_paths <<< "${G2_DEMONSTRATION_DATASETS}"
  for dataset_path in "${dataset_paths[@]}"; do
    if [[ ! -f "${dataset_path}" ]]; then
      echo "demonstration dataset does not exist: ${dataset_path}" >&2
      exit 2
    fi
    dataset_arguments+=(--demonstration-dataset "${dataset_path}")
  done
fi

if [[ ${#dataset_arguments[@]} -eq 0 && "${dataset_cli_present}" != true ]]; then
  echo "preloss_v2 requires the canonical demonstration datasets" >&2
  echo "set G2_DEMONSTRATION_DATASETS or pass repeated --demonstration-dataset" >&2
  exit 2
fi

command=("${repository_root}/scripts/run_g2_recurrent_visual_teacher_supervisor.sh" \
  --recurrent-profile legacy_256x16 \
  --camera-encoder-profile cnn5_160 \
  --num-envs 25 \
  --total-transitions 25000000 \
  --episode-horizon-steps 1600 \
  --learning-starts 1000 \
  --batch-size 64 \
  --replay-capacity 50000 \
  --checkpoint-interval 2000 \
  --random-shift-pad 2 \
  --relative-pose-weight 0.2 \
  --pose-consistency-weight 0.1 \
  --pose-consistency-rotation-weight 0.1 \
  --contact-weight 0.1 \
  --depth-validity-weight 0.05 \
  --student-head-distillation-weight 0.1 \
  --reverse-preroll-max-steps 160 \
  --demonstration-bc-updates 144 \
  --demonstration-batch-size 16 \
  --demonstration-expert-arm-action-scale 0.75 \
  --demonstration-success-augmented-windows 2000 \
  --demonstration-failure-augmented-windows 300 \
  --curriculum-evaluation-envs 5 \
  --her-force-sampling-mixture 0.0 \
  --cross-camera-contrastive-weight 0.0 \
  --temporal-pose-residual-weight 0.0 \
  --scripted-reverse-bootstrap \
  "${dataset_arguments[@]}" \
  "$@")

if [[ "${G2_PRELOSS_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
