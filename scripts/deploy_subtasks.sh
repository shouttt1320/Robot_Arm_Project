#!/usr/bin/env bash
# ==============================================================================
# SO-101 3단계 계층적 자율 주행 배포 런처 (Reach ➔ Grasp ➔ Place)
# ==============================================================================
# 1회 추론 종료 후 자동으로 꺼지지 않고, [Enter] 키로 즉시 재시작할 수 있습니다.
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TOP_CAM=${1:-4}
WRIST_CAM=${2:-6}

# 도움말 요청 시
for arg in "$@"; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
        python3 -m so101_act.deploy_hierarchical --help
        exit 0
    fi
done

trap 'echo -e "\n\n[계층적 배포 세션 종료] 프로그램을 완전히 종료합니다."; exit 0' SIGINT SIGTERM

RUN_COUNT=1

while true; do
    echo "=============================================================================="
    echo " 🤖 [SO-101 3단계 계층적 자율 주행 실행 (FSM Controller) - #${RUN_COUNT}회차]"
    echo "=============================================================================="
    echo " • Stage 1 : Reach (공 상단 접근) ➔ policy_reach.ckpt"
    echo " • Stage 2 : Grasp (공 파지 & 리프트) ➔ policy_grasp.ckpt"
    echo " • Stage 3 : Place (바구니 이동 & 놓기) ➔ policy_place.ckpt"
    echo " • 카메라   : Top(/dev/video${TOP_CAM}), Wrist(/dev/video${WRIST_CAM})"
    echo " • 비상정지 : 카메라 창 [Q] 또는 터미널 [Ctrl+C]"
    echo "=============================================================================="

    python3 -m so101_act.deploy_hierarchical \
        --subtasks-root "${PROJECT_ROOT}/ckpt/so101_subtasks" \
        --top-cam "${TOP_CAM}" \
        --wrist-cam "${WRIST_CAM}" \
        --fps 30 \
        "${@:3}"

    RUN_COUNT=$((RUN_COUNT + 1))

    echo -e "\n------------------------------------------------------------------------------"
    echo -e " 🏁 3단계 자율 주행 1회 사이클이 완료되었습니다. (로봇 홈 복귀 및 토크 해제 완료)"
    echo -e "------------------------------------------------------------------------------"
    read -p " 👉 공을 재배치한 뒤 다시 시작하려면 [Enter], 종료하려면 [q + Enter]: " USER_CHOICE

    if [[ "${USER_CHOICE}" == "q" || "${USER_CHOICE}" == "Q" ]]; then
        echo -e "\n 프로그램을 완전히 종료합니다. 수고하셨습니다!\n"
        break
    fi
    echo -e "\n"
done
