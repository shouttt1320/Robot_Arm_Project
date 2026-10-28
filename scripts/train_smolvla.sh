#!/usr/bin/env bash
# ==============================================================================
# SO-101 SmolVLA (Vision-Language-Action) Fine-tuning Launcher
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

BATCH_SIZE=${1:-8}
STEPS=${2:-20000}
SAVE_FREQ=${3:-5000}

echo "=============================================================================="
echo " 🚀 [SO-101 SmolVLA (~450M) 파인튜닝 시작]"
echo "=============================================================================="
echo " • 베이스 모델 : lerobot/smolvla_base"
echo " • 데이터 경로 : ${PROJECT_ROOT}/data/lerobot_so101_pick_place"
echo " • 체크포인트  : ${PROJECT_ROOT}/ckpt/smolvla_so101"
echo " • 배치 크기   : ${BATCH_SIZE}"
echo " • 총 스텝 수  : ${STEPS} steps"
echo " • 저장 주기   : 매 ${SAVE_FREQ} steps"
echo " • 카메라 매핑 : top -> camera1, wrist -> camera2 (camera3: empty padding)"
echo "=============================================================================="

# LeRobot 가상환경 python 및 CLI 사용
if [ -d "${HOME}/lerobot_env" ]; then
    TRAIN_BIN="${HOME}/lerobot_env/bin/lerobot-train"
else
    TRAIN_BIN="lerobot-train"
fi

"${TRAIN_BIN}" \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.empty_cameras=1 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --dataset.repo_id=local/so101_pick_place \
    --dataset.root="${PROJECT_ROOT}/data/lerobot_so101_pick_place" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --output_dir="${PROJECT_ROOT}/ckpt/smolvla_so101" \
    --job_name=smolvla_so101 \
    --policy.device=cuda \
    "${@:4}"
