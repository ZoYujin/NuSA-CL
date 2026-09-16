#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_LOCATION="${DATA_LOCATION:-${REPO_DIR}/data}"
EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397"
DATASETS=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
FT_LRS=(5e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 5e-5 1e-5 1e-5 1e-5)
read -r -a ADAPTER_PARAMS_ARRAY <<< "${ADAPTER_PARAMS:-q k v o}"

: "${METHOD:?METHOD is required}"
: "${RUN_NAME:?RUN_NAME is required}"

case "${METHOD}" in
  full_ft|lora|milora|inflora|nusa) ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

cd "${REPO_DIR}"
OUTPUT_DIR="${REPO_DIR}/output/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

if ! "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA device:", torch.cuda.get_device_name(0))'; then
  echo "CUDA preflight failed; refusing to run this experiment on CPU." >&2
  exit 1
fi

start_dataset="${START_DATASET:-Aircraft}"
resume_from="${RESUME_FROM:-}"
start_index=-1
for index in "${!DATASETS[@]}"; do
  if [[ "${DATASETS[${index}]}" == "${start_dataset}" ]]; then
    start_index="${index}"
    break
  fi
done
if (( start_index < 0 )); then
  echo "Unknown START_DATASET: ${start_dataset}" >&2
  exit 2
fi
if (( start_index > 0 )) && [[ -z "${resume_from}" ]]; then
  echo "RESUME_FROM is required when START_DATASET is not Aircraft" >&2
  exit 2
fi
if [[ -n "${resume_from}" ]] && [[ ! -f "${resume_from}" ]]; then
  echo "Resume checkpoint does not exist: ${resume_from}" >&2
  exit 2
fi

previous_model="${resume_from}"
for ((index=start_index; index<${#DATASETS[@]}; index++)); do
  dataset="${DATASETS[${index}]}"
  if [[ "${METHOD}" == "full_ft" ]]; then
    lr="${FT_LRS[${index}]}"
  else
    lr="3e-4"
  fi

  echo "==> [$((index + 1))/${#DATASETS[@]}] ${METHOD} on ${dataset} (lr=${lr})"
  args=(
    -u -m src.main
    --method "${METHOD}"
    --train-dataset "${dataset}"
    --eval-datasets "${EVAL_DATASETS}"
    --data-location "${DATA_LOCATION}"
    --batch-size "${BATCH_SIZE:-16}"
    --batch-size-eval "${BATCH_SIZE_EVAL:-128}"
    --num-workers "${SLURM_CPUS_PER_TASK:-8}"
    --iterations "${ITERATIONS:-1000}"
    --lr "${lr}"
    --label-smoothing 0.2
    --seed "${SEED:-42}"
    --save "${OUTPUT_DIR}"
  )

  if [[ -n "${WARMUP_STEPS:-}" ]]; then
    args+=(--warmup-steps "${WARMUP_STEPS}")
  fi

  if [[ "${METHOD}" != "full_ft" ]]; then
    args+=(
      --encoder both
      --position all
      --params "${ADAPTER_PARAMS_ARRAY[@]}"
      --rank 128
      --r-text 128
      --r-vision 128
      --alpha 2
      --dropout 0.25
    )
  fi

  if [[ "${METHOD}" == "nusa" ]]; then
    args+=(--cutoff 0.95)
  fi

  if [[ -n "${previous_model}" ]]; then
    args+=(--load "${previous_model}")
  fi

  "${PYTHON_BIN}" "${args[@]}"
  previous_model="${OUTPUT_DIR}/${dataset}.pth"
done

echo "Completed ${METHOD} MTIL sequence: ${OUTPUT_DIR}"
