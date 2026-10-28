#!/usr/bin/env bash
# ==============================================================================
# SO-101 SmolVLA 멀티모달 LoRA 이어서 학습 (Resume from 5,000 to 20,000 steps)
# ==============================================================================
# • 5,000스텝 상태(가중치 + 옵티마이저)를 그대로 이어받아 20,000스텝까지 완주합니다.
# • 15,000스텝만 추가 학습하므로 약 1시간 15분 만에 완료됩니다. (시간 대폭 절약)
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

STEPS=${1:-20000}
SAVE_FREQ=${2:-5000}

OUTPUT_DIR="${PROJECT_ROOT}/ckpt/smolvla_two_balls_vlm_lora"
CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
LOG_FILE="${PROJECT_ROOT}/train_vlm_lora_resume.log"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "❌ 오류: 이어받을 체크포인트 설정 파일이 없습니다: ${CONFIG_PATH}"
    exit 1
fi

echo "=============================================================================="
echo " 🚀 [SO-101 SmolVLA 멀티모달 LoRA 이어서 학습 (Resume)]"
echo "=============================================================================="
echo " • 이어받을 체크포인트 : ${OUTPUT_DIR}/checkpoints/last"
echo " • 목표 스텝          : ${STEPS} steps"
echo " • 저장 주기          : 매 ${SAVE_FREQ} steps"
echo " • 로그 기록          : ${LOG_FILE}"
echo "=============================================================================="

PYTHON_BIN="/home/jjj/lerobot_env/bin/python3"
TRAIN_BIN="/home/jjj/lerobot_env/bin/lerobot-train"

if [ ! -f "${TRAIN_BIN}" ]; then
    TRAIN_BIN="lerobot-train"
fi

"${TRAIN_BIN}" \
    --config_path="${CONFIG_PATH}" \
    --resume=true \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    "${@:3}" 2>&1 | tee "${LOG_FILE}"

echo -e "\n🎉 이어서 학습 세션이 완료되었습니다. 결과 체크포인트: ${OUTPUT_DIR}"
