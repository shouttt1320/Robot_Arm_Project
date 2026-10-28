#!/usr/bin/env bash
# ==============================================================================
# SO-101 SmolVLA 자연어 지시 기반 자율 추론 및 실로봇 배포 스크립트
# ==============================================================================
# 사용법:
#   1) 기본 명령어로 실행:
#      ./scripts/deploy_smolvla.sh
#
#   2) 원하는 문장 직접 지정하여 실행:
#      ./scripts/deploy_smolvla.sh "Pick up the blue ball and place it in the box"
#      ./scripts/deploy_smolvla.sh "파란 공 상자에 넣어"
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/jjj/lerobot_env/bin/python3"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

# 기본 체크포인트 디렉터리 (two_balls 우선, 없으면 smolvla_so101)
if [ -d "${PROJECT_ROOT}/ckpt/smolvla_two_balls" ]; then
    CKPT_BASE="${PROJECT_ROOT}/ckpt/smolvla_two_balls"
else
    CKPT_BASE="${PROJECT_ROOT}/ckpt/smolvla_so101"
fi

# 1번째 인자: 태스크 문장 (단축어 지원: green, pink, blue)
DEFAULT_TASK="Pick up the green ball and place it in the box"
TASK="${1:-$DEFAULT_TASK}"

if [[ "${TASK}" == "green" || "${TASK}" == "초록" ]]; then
    TASK="Pick up the green ball and place it in the box"
elif [[ "${TASK}" == "pink" || "${TASK}" == "핑크" || "${TASK}" == "분홍" ]]; then
    TASK="Pick up the pink ball and place it in the box"
elif [[ "${TASK}" == "blue" || "${TASK}" == "파랑" ]]; then
    TASK="Pick up the blue ball and place it in the box"
fi

# 2번째 인자: 체크포인트 스텝 번호 또는 경로 (기본값: last = 20,000)
CKPT_ARG="${2:-last}"
EXTRA_ARGS_START=2

# 만약 2번째 인자가 숫자인 경우 (예: 5000, 10000, 15000, 20000)
if [[ "${CKPT_ARG}" =~ ^[0-9]+$ ]]; then
    STEP_PADDED=$(printf "%06d" "${CKPT_ARG}")
    CHECKPOINT_PATH="${CKPT_BASE}/checkpoints/${STEP_PADDED}/pretrained_model"
    EXTRA_ARGS_START=3
elif [[ "${CKPT_ARG}" == "last" || "${CKPT_ARG}" == "final" ]]; then
    CHECKPOINT_PATH="${CKPT_BASE}/checkpoints/last/pretrained_model"
    EXTRA_ARGS_START=3
elif [[ -d "${CKPT_ARG}" ]]; then
    CHECKPOINT_PATH="${CKPT_ARG}"
    EXTRA_ARGS_START=3
elif [[ "${CKPT_ARG}" == --* ]]; then
    CHECKPOINT_PATH="${CKPT_BASE}/checkpoints/last/pretrained_model"
    EXTRA_ARGS_START=2
else
    CHECKPOINT_PATH="${CKPT_BASE}/checkpoints/last/pretrained_model"
fi

if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "❌ 오류: 지정한 체크포인트 디렉터리를 찾을 수 없습니다: ${CHECKPOINT_PATH}"
    echo "💡 사용 가능한 스텝: 5000, 10000, 15000, 20000 (또는 last)"
    exit 1
fi

# 카메라 디바이스 자동 감지 (Top: PC Cam, Wrist: Innomaker)
CAM_DETECT=$(${PYTHON_BIN} -c "from hardware.camera_interface import auto_detect_camera_indices; t, w = auto_detect_camera_indices(); print(f'{t} {w}')" 2>/dev/null)
read -r TOP_CAM WRIST_CAM <<< "${CAM_DETECT:-4 7}"
MAX_STEPS=600

# 도움말 요청 시
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    ${PYTHON_BIN} -m models.so101_vla.deploy_smolvla --help
    exit 0
fi

trap 'echo -e "\n\n[배포 세션 종료] 프로그램을 완전히 종료합니다."; exit 0' SIGINT SIGTERM

RUN_COUNT=1

while true; do
    echo "=============================================================================="
    echo " 🤖 [SO-101 SmolVLA 자연어 기반 자율 구동 - #${RUN_COUNT}회차]"
    echo "=============================================================================="
    echo " • 체크포인트 : ${CHECKPOINT_PATH}"
    echo " • 자연어 명령 : \"${TASK}\""
    echo " • 카메라 장치 : Top(/dev/video${TOP_CAM}), Wrist(/dev/video${WRIST_CAM})"
    echo " • 최대 스텝   : ${MAX_STEPS}스텝 (약 20초)"
    echo " • 정지 방법   : 프리뷰 화면에서 [Q] 또는 터미널 [Ctrl+C]"
    echo "=============================================================================="

    PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/upstream_models/lerobot/src" \
    ${PYTHON_BIN} -m models.so101_vla.deploy_smolvla \
        --checkpoint "${CHECKPOINT_PATH}" \
        --task "${TASK}" \
        --top-cam "${TOP_CAM}" \
        --wrist-cam "${WRIST_CAM}" \
        --max-steps "${MAX_STEPS}" \
        "${@:${EXTRA_ARGS_START}}"

    RUN_COUNT=$((RUN_COUNT + 1))

    echo -e "\n------------------------------------------------------------------------------"
    echo -e " 🏁 자율 제어 1회 사이클이 완료되었습니다. (홈 복귀 및 모터 안전 정지 완료)"
    echo -e "------------------------------------------------------------------------------"
    echo -e " 옵션:"
    echo -e "  [Enter]         : 동일한 명령(\"${TASK}\")으로 다시 실행"
    echo -e "  [새 문장 입력] : 다음 회차에 실행할 새 자연어 명령 입력 후 Enter"
    echo -e "  [q + Enter]     : 프로그램 완전 종료"
    echo -e "------------------------------------------------------------------------------"
    read -p " 👉 선택하세요: " USER_INPUT

    if [[ "${USER_INPUT}" == "q" || "${USER_INPUT}" == "Q" ]]; then
        echo -e "\n 프로그램을 종료합니다. 수고하셨습니다!\n"
        break
    elif [[ -n "${USER_INPUT}" ]]; then
        TASK="${USER_INPUT}"
        echo -e "\n 🔄 새로운 명령으로 설정되었습니다: \"${TASK}\""
    fi
    echo -e "\n"
done
