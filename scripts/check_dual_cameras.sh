#!/usr/bin/env bash
# 양안 카메라 실시간 시점 및 초점 고정(Auto Focus Lock) 점검 스크립트
# 기본값: left=4, right=6, focus=25 (오토포커스 비활성화 고정)

LEFT_ID=${1:-4}
RIGHT_ID=${2:-6}
FOCUS_VAL=${3:-25}

echo "=== 양안(Dual Eyes) 카메라 시점 및 초점(Focus Lock) 점검 실행 ==="
echo "LEFT: /dev/video${LEFT_ID}, RIGHT: /dev/video${RIGHT_ID}, FOCUS: ${FOCUS_VAL} (고정)"
echo "단축키: [f] 원샷 오토포커스 후 고정 | [+/-] 초점 미세조정 | [s] 좌우 스왑 | [c] 캡처 | [q] 종료"
echo ""

python3 hardware/check_dual_cameras.py --left_id "${LEFT_ID}" --right_id "${RIGHT_ID}" --focus "${FOCUS_VAL}" "${@:4}"
