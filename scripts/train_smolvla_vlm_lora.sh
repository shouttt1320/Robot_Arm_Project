#!/usr/bin/env bash
# ==============================================================================
# SO-101 SmolVLA 진정한 멀티모달 LoRA 파인튜닝 스크립트 (VLM + Expert)
# ==============================================================================
# • 타겟 모듈 : SigLIP 비전 타워 + SmolLM2 언어 모델 + Action Expert + 프로젝션 전체
# • 베이스    : lerobot/smolvla_base (450M)
# • 데이터셋  : Green 51개 + Pink 51개 (총 102개 에피소드)
# • 파라미터  : LoRA rank 16 (학습 파라미터 약 2.09M, 전체의 0.46%)
# • 예상 VRAM : 약 4.2GB (RTX 5050 8GB VRAM 내 여유 구동)
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# VRAM 단편화 방지 및 8GB VRAM 최적화
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

BATCH_SIZE=${1:-4}
STEPS=${2:-20000}
SAVE_FREQ=${3:-5000}
EVAL_FREQ=${4:-500}

DATA_DIR="${PROJECT_ROOT}/data/lerobot_so101_two_balls"
OUTPUT_DIR="${PROJECT_ROOT}/ckpt/smolvla_two_balls_vlm_lora"
LOG_FILE="${PROJECT_ROOT}/train_vlm_lora.log"

# VLM(비전 + 텍스트) + Action Expert 전체를 포괄하는 LoRA 타겟 정규식
LORA_TARGETS=".*(vlm.*(q|v)_proj|lm_expert.*(q|v)_proj|state_proj|action_in_proj|action_out_proj).*"

echo "=============================================================================="
echo " 🚀 [SO-101 SmolVLA 진정한 멀티모달 LoRA (Vision + Language + Expert)]"
echo "=============================================================================="
echo " • 베이스 모델 : lerobot/smolvla_base"
echo " • 데이터셋   : ${DATA_DIR} (102 에피소드)"
echo " • 출력 폴더   : ${OUTPUT_DIR}"
echo " • LoRA 타겟   : SigLIP(비전) + SmolLM2(언어) + Action Expert 전체"
echo " • 배치 크기   : ${BATCH_SIZE}"
echo " • 총 스텝 수  : ${STEPS} steps"
echo " • 저장 주기   : 매 ${SAVE_FREQ} steps"
echo " • 로그 기록   : ${LOG_FILE}"
echo "=============================================================================="

PYTHON_BIN="/home/jjj/lerobot_env/bin/python3"
TRAIN_BIN="/home/jjj/lerobot_env/bin/lerobot-train"

if [ ! -f "${TRAIN_BIN}" ]; then
    TRAIN_BIN="lerobot-train"
fi

# 학습 실행 (tee 로 콘솔과 파일 동시 기록)
"${TRAIN_BIN}" \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --dataset.repo_id=local/so101_two_balls \
    --dataset.root="${DATA_DIR}" \
    --dataset.eval_split=0.0 \
    --eval_steps=0 \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=smolvla_two_balls_vlm_lora \
    --policy.device=cuda \
    --peft.target_modules="${LORA_TARGETS}" \
    --peft.r=16 \
    --peft.lora_alpha=16 \
    --peft.method_type=LORA \
    "${@:5}" 2>&1 | tee "${LOG_FILE}"

echo -e "\n🎉 학습 세션이 완료되었습니다. 결과 체크포인트: ${OUTPUT_DIR}"
