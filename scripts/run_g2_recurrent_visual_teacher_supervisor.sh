#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${G2_VISUAL_TEACHER_PYTHON:-/data/fain-data/test/genie_sim_isaaclab3_sim601/bin/python}"
output_root="${G2_VISUAL_TEACHER_OUTPUT_ROOT:-/data/fain-data/test/output}"
timestamp="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${output_root}"

exec "${python_bin}" \
  "${repository_root}/scripts/diagnostics/run_g2_recurrent_visual_teacher_supervisor.py" \
  --python "${python_bin}" \
  --output "${output_root}/g2_recurrent_visual_teacher_${timestamp}" \
  "$@"
