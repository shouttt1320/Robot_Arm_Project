#!/usr/bin/env bash
# ==============================================================================
# SO-101 양안(Dual Eyes) DINOv2 ACT 실시간 자율 추론 실행기
# ==============================================================================
# 사용법:
#   bash scripts/deploy_dual_eyes_act.sh
#   bash scripts/deploy_dual_eyes_act.sh --max-steps 1200
#   bash scripts/deploy_dual_eyes_act.sh --chunk 20 --max-steps 900
#   bash scripts/deploy_dual_eyes_act.sh --ckpt policy_best.pth
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 도움말 요청 시
for arg in "$@"; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
        python3 -m models.so101_dual_dinov2_act.deploy_real_robot --help
        exit 0
    fi
done

# 기본값
LEFT_CAM=4
RIGHT_CAM=6
FOCUS_VAL=25
MAX_STEPS=600

# 인자에서 --max-steps 추출 (배너 표시용)
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--max-steps" && $((i+1)) -lt ${#ARGS[@]} ]]; then
        MAX_STEPS="${ARGS[i+1]}"
    fi
done

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# 하드웨어 권한 점검
sudo chmod 666 /dev/ttyACM* 2>/dev/null || true

# Ctrl+C 시 안전하게 완전 종료
trap 'echo -e "\n\n${YELLOW}[배포 세션 종료] 프로그램을 완전히 종료합니다.${NC}"; exit 0' SIGINT SIGTERM

RUN_COUNT=1

while true; do
    echo -e "${CYAN}==============================================================================${NC}"
    echo -e "${BOLD} 🤖 [SO-101 양안 DINOv2 ACT 실시간 자율 구동 - #${RUN_COUNT}회차]${NC}"
    echo -e "${CYAN}==============================================================================${NC}"
    echo -e " • 모델 경로  : ${GREEN}${PROJECT_ROOT}/ckpt/so101_dual_dinov2_act/${NC}"
    echo -e " • 좌안 카메라: ${GREEN}/dev/video${LEFT_CAM}${NC} (Left Eye)"
    echo -e " • 우안 카메라: ${GREEN}/dev/video${RIGHT_CAM}${NC} (Right Eye)"
    echo -e " • 초점 고정  : ${GREEN}Focus = ${FOCUS_VAL} (오토포커스 OFF)${NC}"
    echo -e " • 최대 스텝  : ${YELLOW}${MAX_STEPS}스텝 (약 $((MAX_STEPS / 30))초)${NC}"
    echo -e " • 1회 정지   : 프리뷰 창에서 [Q] 또는 터미널 [Ctrl+C]"
    echo -e "${CYAN}==============================================================================${NC}"

    python3 -m models.so101_dual_dinov2_act.deploy_real_robot \
        --left-cam "${LEFT_CAM}" \
        --right-cam "${RIGHT_CAM}" \
        --focus "${FOCUS_VAL}" \
        --max-steps "${MAX_STEPS}" \
        "$@"

    RUN_COUNT=$((RUN_COUNT + 1))

    echo -e "\n${CYAN}------------------------------------------------------------------------------${NC}"
    echo -e " 🏁 자율 주행 1회 사이클 완료! (로봇 홈 복귀 및 안전 대기 완료)"
    echo -e "${CYAN}------------------------------------------------------------------------------${NC}"
    read -p " 👉 물체를 재배치한 뒤 다시 시작하려면 [Enter], 종료하려면 [q + Enter]: " USER_CHOICE

    if [[ "${USER_CHOICE}" == "q" || "${USER_CHOICE}" == "Q" ]]; then
        echo -e "\n 프로그램을 완전히 종료합니다. 수고하셨습니다!\n"
        break
    fi
    echo -e "\n"
done
