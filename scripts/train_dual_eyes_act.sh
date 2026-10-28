#!/usr/bin/env bash
# ==============================================================================
# SO-101 양안(Dual Eyes) DINOv2 ACT 모델 학습 실행 스크립트
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 기본 하이퍼파라미터
DATA_DIR="data/so101_dual_eyes"
CKPT_DIR="ckpt/so101_dual_dinov2_act"
BATCH_SIZE=${1:-8}
NUM_EPOCHS=${2:-5000}
SAVE_EVERY=${3:-1000}
LR="1e-5"

echo "=============================================================================="
echo "   🚀 SO-101 양안(Dual Eyes) DINOv2 ACT 모델 학습 시작"
echo "=============================================================================="
echo " 📁 데이터 경로   : ${DATA_DIR}"
echo " 💾 체크포인트    : ${CKPT_DIR}"
echo " 📦 배치 크기     : ${BATCH_SIZE}"
echo " 🔁 총 에포크     : ${NUM_EPOCHS}"
echo " ⏱️ 저장 주기     : ${SAVE_EVERY} 에포크마다"
echo " 🧠 이미지 백본   : DINOv2 ViT-S/14 (Frozen)"
echo " 👁️ 카메라 구성   : cam_left (/dev/video4) + cam_right (/dev/video6)"
echo "=============================================================================="

# Python 학습 실행
python3 -m models.so101_dual_dinov2_act.train \
    --data_dir "${DATA_DIR}" \
    --ckpt_dir "${CKPT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --save_every "${SAVE_EVERY}" \
    --lr "${LR}" \
    "${@:4}"
