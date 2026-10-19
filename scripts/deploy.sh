#!/usr/bin/env bash
# ==============================================================================
# SO-101 Pick-and-Place Autonomous Deployment (Phase 6)
# ==============================================================================
# 1회 추론 종료 후 자동으로 꺼지지 않고, [Enter] 키로 즉시 재시작할 수 있습니다.
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TOP_CAM=4
WRIST_CAM=6
MAX_STEPS=600

# 도움말 요청 시
for arg in "$@"; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
        /usr/bin/python3 -m so101_act.deploy_real_robot --help
        exit 0
    fi
done

# Ctrl+C 시 안전하게 완전 종료
trap 'echo -e "\n\n[배포 세션 종료] 프로그램을 완전히 종료합니다."; exit 0' SIGINT SIGTERM

RUN_COUNT=1

while true; do
    echo "=============================================================================="
    echo " 🤖 [SO-101 실시간 완전 자율 구동 - #${RUN_COUNT}회차]"
    echo "=============================================================================="
    echo " • 모델 가중치: ${PROJECT_ROOT}/ckpt/so101_act/policy_best.ckpt"
    echo " • 카메라 장치: Top(/dev/video${TOP_CAM}), Wrist(/dev/video${WRIST_CAM})"
    echo " • 최대 스텝  : ${MAX_STEPS}스텝 (약 20초)"
    echo " • 1회 정지   : 프리뷰 창에서 [Q] 또는 터미널 [Ctrl+C]"
    echo "=============================================================================="

    /usr/bin/python3 -m so101_act.deploy_real_robot \
        --top-cam "${TOP_CAM}" \
        --wrist-cam "${WRIST_CAM}" \
        --max-steps "${MAX_STEPS}" \
        "$@"

    RUN_COUNT=$((RUN_COUNT + 1))

    echo -e "\n------------------------------------------------------------------------------"
    echo -e " 🏁 자율 주행 1회 사이클이 완료되었습니다. (로봇 홈 복귀 및 토크 해제 완료)"
    echo -e "------------------------------------------------------------------------------"
    read -p " 👉 공을 재배치한 뒤 다시 시작하려면 [Enter], 종료하려면 [q + Enter]: " USER_CHOICE

    if [[ "${USER_CHOICE}" == "q" || "${USER_CHOICE}" == "Q" ]]; then
        echo -e "\n 프로그램을 완전히 종료합니다. 수고하셨습니다!\n"
        break
    fi
    echo -e "\n"
done
