#!/usr/bin/env bash
# ==============================================================================
# SO-101 SmolVLA 그린/핑크 멀티태스크 학습 & Train/Val 실시간 검증 런처
# ==============================================================================
# • 데이터셋   : Green 26개 + Pink 27개 (총 53개 에피소드)
# • Train/Val  : 80% Train (43개) / 20% Held-out Validation (10개)
# • 검증 주기  : 500 스텝마다 Unseen Validation Loss 자동 측정
# • 로그 저장  : train_multitask.log 에 전 과정 실시간 저장
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

BATCH_SIZE=${1:-8}
STEPS=${2:-20000}
SAVE_FREQ=${3:-5000}
EVAL_FREQ=${4:-500}

DATA_DIR="${PROJECT_ROOT}/data/lerobot_so101_two_balls"
OUTPUT_DIR="${PROJECT_ROOT}/ckpt/smolvla_two_balls"
LOG_FILE="${PROJECT_ROOT}/train_multitask.log"

echo "=============================================================================="
echo " 🚀 [SO-101 SmolVLA 멀티태스크 파인튜닝 (Green vs Pink)]"
echo "=============================================================================="
echo " • 베이스 모델 : lerobot/smolvla_base"
echo " • 데이터셋   : ${DATA_DIR}"
echo " • 출력 폴더   : ${OUTPUT_DIR}"
echo " • 배치 크기   : ${BATCH_SIZE}"
echo " • 총 스텝 수  : ${STEPS} steps"
echo " • 검증 분할   : 20% (Held-out Validation, 약 10개 에피소드)"
echo " • 검증 주기   : 매 ${EVAL_FREQ} steps (Val Loss 자동 계산)"
echo " • 저장 주기   : 매 ${SAVE_FREQ} steps"
echo " • 로그 기록   : ${LOG_FILE}"
echo "=============================================================================="

PYTHON_BIN="/home/jjj/lerobot_env/bin/python3"
TRAIN_BIN="/home/jjj/lerobot_env/bin/lerobot-train"

if [ ! -f "${TRAIN_BIN}" ]; then
    TRAIN_BIN="lerobot-train"
fi

# 학습 실행 및 콘솔 + 파일 동시 기록 (tee)
"${TRAIN_BIN}" \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.empty_cameras=1 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --dataset.repo_id=local/so101_two_balls \
    --dataset.root="${DATA_DIR}" \
    --dataset.eval_split=0.0 \
    --eval_steps=0 \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=smolvla_two_balls \
    --policy.device=cuda \
    "${@:5}" 2>&1 | tee "${LOG_FILE}"

echo -e "\n🎉 학습 세션이 완료되었습니다. 결과 체크포인트: ${OUTPUT_DIR}"
