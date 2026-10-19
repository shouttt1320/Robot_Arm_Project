#!/usr/bin/env bash
# ==============================================================================
# SO-101 Pick-and-Place Teleoperation & Data Collection Launcher (Phase 4)
# ==============================================================================
# 이 스크립트는 텔레오퍼레이션과 HDF5 데이터 녹화를 원클릭으로 실행합니다.
# record_episodes.py 내부에 실시간 텔레옵(1:1 추종)이 이미 내장되어 있으므로
# 별도의 teleop.py를 동시에 실행하지 않아도 안전하게 조작과 녹화가 동시에 진행됩니다.
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TASK_NAME="so101_pick_place"
TOP_CAM=4
WRIST_CAM=6
FPS=30
MAX_SEC=15

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# 도움말 요청 시 파이썬 도움말 바로 출력
for arg in "$@"; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
        /usr/bin/python3 -m collection.record_episodes --help
        exit 0
    fi
done

if [ -t 1 ]; then
    clear || true
fi

echo -e "${CYAN}==============================================================================${NC}"
echo -e "${BOLD}   🦾 SO-101 픽앤플레이스 실시간 텔레옵 & 데이터 수집 통합 실행기 (Phase 4)${NC}"
echo -e "${CYAN}==============================================================================${NC}"
echo -e " 📌 태스크 이름   : ${GREEN}${TASK_NAME}${NC}"
echo -e " 🎥 외부 카메라   : ${GREEN}/dev/video${TOP_CAM}${NC} (Top View)"
echo -e " 🎥 손목 카메라   : ${GREEN}/dev/video${WRIST_CAM}${NC} (Wrist View)"
echo -e " ⏱️ 수집 시간     : ${GREEN}시간 제한 없음${NC} (스페이스바로 자유롭게 시작/종료 제어)"
echo -e " 📁 데이터 저장   : ${YELLOW}${PROJECT_ROOT}/data/${TASK_NAME}/${NC}"
echo -e "${CYAN}------------------------------------------------------------------------------${NC}"

# 1. 하드웨어 사전 점검 (Pre-flight Check)
echo -e "${BOLD}[1/3] 하드웨어 및 장치 연결 상태 확인 중...${NC}"

# 시리얼 포트 확인
if [ ! -e "/dev/ttyACM0" ] || [ ! -e "/dev/ttyACM1" ]; then
    echo -e "${RED}[오류] 로봇 USB 시리얼 포트(/dev/ttyACM0, /dev/ttyACM1)가 감지되지 않습니다!${NC}"
    echo -e "       USB 케이블 연결을 확인하고 다시 실행하세요."
    exit 1
fi
echo -e "  ✅ 로봇 시리얼 포트 확인 완료 (/dev/ttyACM0, /dev/ttyACM1)"

# 시리얼 권한 확인
if [ ! -r "/dev/ttyACM0" ] || [ ! -w "/dev/ttyACM0" ]; then
    echo -e "${YELLOW}[경고] /dev/ttyACM0 읽기/쓰기 권한이 없습니다. sudo 권한 부여를 시도합니다...${NC}"
    sudo chmod 666 /dev/ttyACM* 2>/dev/null || true
fi

# 카메라 확인
if [ ! -e "/dev/video${TOP_CAM}" ]; then
    echo -e "${YELLOW}[경고] Top 카메라(/dev/video${TOP_CAM})를 찾을 수 없습니다.${NC}"
fi
if [ ! -e "/dev/video${WRIST_CAM}" ]; then
    echo -e "${YELLOW}[경고] Wrist 카메라(/dev/video${WRIST_CAM})를 찾을 수 없습니다.${NC}"
fi
echo -e "  ✅ 카메라 장치 점검 완료"

# 캘리브레이션 파일 확인
if [ ! -f "configs/calibration.json" ] && [ ! -f "calibration.json" ]; then
    echo -e "${RED}[오류] calibration.json 파일이 없습니다. 먼저 'python3 -m hardware.calibrate'를 실행하세요.${NC}"
    exit 1
fi
echo -e "  ✅ 로봇 캘리브레이션 설정 파일 확인 완료 (configs/calibration.json)"

echo -e "\n${BOLD}[2/3] 조작 및 녹화 가이드 (스페이스바 제어)${NC}"
echo -e "  1. 실행되면 ${GREEN}실시간 텔레오퍼레이션이 즉시 활성화${NC}됩니다. (리더 팔을 움직여 로봇을 초기 준비 자세로 이동)"
echo -e "  2. 준비가 되면 카메라 창 또는 터미널에서 ${BOLD}[스페이스바]${NC}를 누르세요. (녹화 즉시 시작!)"
echo -e "  3. 시간 제한 없이 여유 있게 물체를 집어 옮긴 뒤 ${BOLD}[스페이스바]${NC}를 다시 누르면 녹화 종료!"
echo -e "  4. 저장 시 ${BOLD}[스페이스바/Y]${NC}, 실수 시 ${BOLD}[n]${NC}을 누르면 해당 에피소드가 폐기/재촬영됩니다."
echo -e "  5. 중단하거나 완전히 종료하려면 ${BOLD}[Ctrl + C]${NC} 또는 ${BOLD}[q]${NC}를 입력하세요."
echo -e "${CYAN}------------------------------------------------------------------------------${NC}"

# 현재 수집된 에피소드 개수 확인
EXISTING_COUNT=$(ls data/${TASK_NAME}/episode_*.hdf5 2>/dev/null | wc -l || echo 0)
echo -e " 📊 현재까지 저장된 에피소드: ${GREEN}${EXISTING_COUNT}개${NC} (목표: 30~50개)"
echo -e "${CYAN}------------------------------------------------------------------------------${NC}"

if [ -t 0 ]; then
    read -p "👉 텔레옵 및 데이터 수집을 시작하시겠습니까? [Enter/Space: 시작, Ctrl+C: 취소] "
fi

echo -e "\n${BOLD}[3/3] 통합 텔레옵 및 데이터 수집 프로세스 구동 중...${NC}\n"

# Python 스크립트 실행
/usr/bin/python3 -m collection.record_episodes \
    --task "${TASK_NAME}" \
    --top-cam "${TOP_CAM}" \
    --wrist-cam "${WRIST_CAM}" \
    --fps "${FPS}" \
    "$@"

# 종료 후 결과 요약 및 검증 안내
echo -e "\n${CYAN}==============================================================================${NC}"
FINAL_COUNT=$(ls data/${TASK_NAME}/episode_*.hdf5 2>/dev/null | wc -l || echo 0)
echo -e " 🏁 데이터 수집 세션이 종료되었습니다. (총 저장된 에피소드: ${GREEN}${FINAL_COUNT}개${NC})"
echo -e "${CYAN}==============================================================================${NC}"

if [ "${FINAL_COUNT}" -gt 0 ] && [ -t 0 ]; then
    echo -e " 🔍 수집된 데이터를 바로 검증하시겠습니까?"
    echo -e "    1) 방금 녹화된 마지막 에피소드 검증 및 영상(MP4)/그래프(PNG) 생성"
    echo -e "    2) 전체 에피소드 무결성 일괄 점검 리포트"
    echo -e "    3) 종료"
    read -p " 선택 [1/2/3]: " CHOICE
    case $CHOICE in
        1)
            LAST_IDX=$((FINAL_COUNT - 1))
            python3 -m collection.visualize_data --task "${TASK_NAME}" --episode "${LAST_IDX}"
            ;;
        2)
            python3 -m collection.visualize_data --task "${TASK_NAME}" --all
            ;;
        *)
            echo "프로그램을 종료합니다."
            ;;
    esac
fi
