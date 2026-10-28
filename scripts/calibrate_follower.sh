#!/usr/bin/env bash
# ==============================================================================
# SO-101 Follower(팔로워) 로봇 팔 단독 캘리브레이션 실행기
# ==============================================================================
# - 리더 팔 연결 불필요 (Leader 설정은 100% 안전 보존)
# - 이전 영점 대비 모터 축 틀어짐(편차) 실시간 모니터링
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

FOLLOWER_PORT=${1:-/dev/ttyACM0}

echo "=== SO-101 Follower 단독 캘리브레이션 시작 ==="
echo "팔로워 포트: ${FOLLOWER_PORT}"
echo ""

python3 -m hardware.calibrate --follower "${FOLLOWER_PORT}" --follower-only "${@:2}"
