#!/usr/bin/env bash
# ==============================================================================
# SO-101 Pick-and-Place ACT Training Launcher (Phase 5)
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 도움말 요청 시
for arg in "$@"; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
        /usr/bin/python3 -m so101_act.train --help
        exit 0
    fi
done

echo "=============================================================================="
echo " 🚀 [SO-101 ACT 모델 학습 시작 (Phase 5)]"
echo "=============================================================================="
echo " • 대상 태스크: so101_pick_place"
echo " • 데이터 경로: ${PROJECT_ROOT}/data/so101_pick_place"
echo " • 저장 경로  : ${PROJECT_ROOT}/ckpt/so101_act"
echo "=============================================================================="

/usr/bin/python3 -m so101_act.train "$@"
