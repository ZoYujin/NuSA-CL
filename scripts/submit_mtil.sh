#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {full_ft|lora|milora|inflora|nusa} [run_name]" >&2
  exit 2
fi

METHOD="$1"
case "${METHOD}" in
  full_ft|lora|milora|inflora|nusa) ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
RUN_NAME="${2:-${METHOD}_seed${SEED:-42}}"
OUTPUT_DIR="${REPO_DIR}/output/${RUN_NAME}"
LOG_DIR="${REPO_DIR}/logs/${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

exclude_args=()
if [[ -n "${EXCLUDE_NODES:-}" ]]; then
  exclude_args=(--exclude="${EXCLUDE_NODES}")
fi

job_id=$(sbatch --parsable \
  --partition="${PARTITION:-P2}" \
  --gres=gpu:1 \
  --cpus-per-task="${CPUS_PER_TASK:-8}" \
  --mem="${MEMORY:-50G}" \
  --time="${TIME_LIMIT:-12:00:00}" \
  --job-name="${METHOD}-mtil" \
  --output="${LOG_DIR}/${METHOD}-%j.out" \
  --error="${LOG_DIR}/${METHOD}-%j.err" \
  "${exclude_args[@]}" \
  --export="ALL,METHOD=${METHOD},RUN_NAME=${RUN_NAME},DATA_LOCATION=${DATA_LOCATION:-${REPO_DIR}/data},ITERATIONS=${ITERATIONS:-1000},SEED=${SEED:-42},BATCH_SIZE=${BATCH_SIZE:-16},BATCH_SIZE_EVAL=${BATCH_SIZE_EVAL:-128},WARMUP_STEPS=${WARMUP_STEPS:-},ADAPTER_PARAMS=${ADAPTER_PARAMS:-q k v o},START_DATASET=${START_DATASET:-Aircraft},RESUME_FROM=${RESUME_FROM:-}" \
  "${REPO_DIR}/scripts/run_mtil.sh")

echo "${METHOD} MTIL job: ${job_id}"
echo "log directory: ${LOG_DIR}"
echo "output directory: ${OUTPUT_DIR}"
