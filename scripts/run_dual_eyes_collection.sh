#!/usr/bin/env bash
# ==============================================================================
# SO-101 양안(Dual Eyes) 카메라 텔레옵 & 데이터 수집 실행기
# ==============================================================================
# - 좌안: /dev/video4, 우안: /dev/video6
# - 오토포커스 비활성화 및 초점 25 단단히 고정 (Focus Lock)
# - 저장 경로: data/so101_dual_eyes/ (DINOv2 ACT 학습용 HDF5)
# - 스페이스바 제어 (시작/종료)
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TASK_NAME="so101_dual_eyes"
LEFT_CAM=${1:-4}
RIGHT_CAM=${2:-6}
FOCUS_VAL=${3:-25}
FPS=30

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}==============================================================================${NC}"
echo -e "${BOLD}   🦾 SO-101 양안(Dual Eyes) 실시간 텔레옵 & 에피소드 데이터 수집기${NC}"
echo -e "${CYAN}==============================================================================${NC}"
echo -e " 📌 태스크 이름   : ${GREEN}${TASK_NAME}${NC}"
echo -e " 👁️ 좌안 카메라   : ${GREEN}/dev/video${LEFT_CAM}${NC} (Left Eye)"
echo -e " 👁️ 우안 카메라   : ${GREEN}/dev/video${RIGHT_CAM}${NC} (Right Eye)"
echo -e " 🔒 초점 고정     : ${GREEN}Focus = ${FOCUS_VAL} (오토포커스 OFF / 고정)${NC}"
echo -e " ⏱️ 수집 FPS      : ${GREEN}${FPS} FPS${NC}"
echo -e " 📁 데이터 저장   : ${YELLOW}${PROJECT_ROOT}/data/${TASK_NAME}/${NC}"
echo -e "${CYAN}------------------------------------------------------------------------------${NC}"

# 하드웨어 점검
if [ ! -e "/dev/ttyACM0" ] && [ ! -e "/dev/ttyACM1" ]; then
    echo -e "${RED}[오류] 로봇 USB 시리얼 포트(/dev/ttyACM*)가 감지되지 않습니다!${NC}"
    exit 1
fi

sudo chmod 666 /dev/ttyACM* 2>/dev/null || true

# 현재 수집된 에피소드 수
EXISTING_COUNT=$(ls data/${TASK_NAME}/episode_*.hdf5 2>/dev/null | wc -l || echo 0)
echo -e " 📊 현재까지 저장된 에피소드: ${GREEN}${EXISTING_COUNT}개${NC} (추천: 30~50개)"
echo -e "${CYAN}------------------------------------------------------------------------------${NC}"
echo -e " ⌨️ [스페이스바] 한 번 누르면 녹화 시작 ➔ 작업 완료 후 다시 누르면 녹화 종료!"
echo -e " 💾 저장 여부: [스페이스바/Y] 저장, [n] 폐기 및 재촬영, [q] 종료"
echo -e "${CYAN}------------------------------------------------------------------------------${NC}"

if [ -t 0 ]; then
    read -p "👉 텔레옵 및 데이터 수집을 시작하시겠습니까? [Enter/Space: 시작, Ctrl+C: 취소] "
fi

echo -e "\n${BOLD}🚀 데이터 수집 프로세스 실행 중...${NC}\n"

python3 -m collection.record_episodes \
    --task "${TASK_NAME}" \
    --left-cam "${LEFT_CAM}" \
    --right-cam "${RIGHT_CAM}" \
    --focus "${FOCUS_VAL}" \
    --fps "${FPS}" \
    "${@:4}"
