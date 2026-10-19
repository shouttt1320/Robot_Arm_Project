#!/usr/bin/env bash
# ==============================================================================
# SO-101 픽앤플레이스 서브태스크 3단계 순차 학습 스크립트
# ==============================================================================
# 1. Reach (접근/이동) : 홈 -> 공 상단 도달
# 2. Grasp (잡기)      : 공 상단 하강 -> 파지 -> 리프트
# 3. Place (놓기)      : 공 이동 -> 바구니 개방 -> 복귀
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

EPOCHS=${1:-100}
BATCH_SIZE=${2:-8}
LR=${3:-1e-4}

echo "=============================================================================="
echo " 🚀 [SO-101 3단계 서브태스크 순차 학습 시작 (Reach ➔ Grasp ➔ Place)]"
echo "=============================================================================="
echo " • 목표 에포크 (에피소드당) : ${EPOCHS} epochs"
echo " • 배치 크기                : ${BATCH_SIZE}"
echo " • 학습률                   : ${LR}"
echo "=============================================================================="

# 1. Reach (이동) 학습
echo -e "\n------------------------------------------------------------------------------"
echo " 🟢 [1/3] Step 1: Reach (접근/이동) 모델 학습 중..."
echo "------------------------------------------------------------------------------"
python3 -m so101_act.train \
    --data-dir "${PROJECT_ROOT}/data/so101_subtasks/reach" \
    --ckpt-dir "${PROJECT_ROOT}/ckpt/so101_subtasks/reach" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --chunk-size 50

# 2. Grasp (잡기) 학습
echo -e "\n------------------------------------------------------------------------------"
echo " 🟡 [2/3] Step 2: Grasp (잡기 & 리프트) 모델 학습 중..."
echo "------------------------------------------------------------------------------"
python3 -m so101_act.train \
    --data-dir "${PROJECT_ROOT}/data/so101_subtasks/grasp" \
    --ckpt-dir "${PROJECT_ROOT}/ckpt/so101_subtasks/grasp" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --chunk-size 25

# 3. Place (놓기) 학습
echo -e "\n------------------------------------------------------------------------------"
echo " 🔵 [3/3] Step 3: Place (이동 & 바구니 놓기) 모델 학습 중..."
echo "------------------------------------------------------------------------------"
python3 -m so101_act.train \
    --data-dir "${PROJECT_ROOT}/data/so101_subtasks/place" \
    --ckpt-dir "${PROJECT_ROOT}/ckpt/so101_subtasks/place" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --chunk-size 40

echo -e "\n=============================================================================="
echo " 🎉 [모든 3단계 서브태스크 학습 완료!]"
echo " • Reach 모델 : ${PROJECT_ROOT}/ckpt/so101_subtasks/reach/policy_best.ckpt"
echo " • Grasp 모델 : ${PROJECT_ROOT}/ckpt/so101_subtasks/grasp/policy_best.ckpt"
echo " • Place 모델 : ${PROJECT_ROOT}/ckpt/so101_subtasks/place/policy_best.ckpt"
echo " 👉 추론 실행 : ./scripts/deploy_subtasks.sh"
echo "=============================================================================="
