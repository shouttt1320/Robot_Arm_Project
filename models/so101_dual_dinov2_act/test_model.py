"""양안 DINOv2 ACT 모델 단위 테스트 및 입출력 Shape 검증 스크립트."""

import sys
import os
import torch

# 프로젝트 루트 경로 추가
cur = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(cur))
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "models"))
    sys.path.insert(0, project_root)

from so101_dual_dinov2_act.config import (
    BACKBONE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    HIDDEN_DIM,
    IMAGE_SIZE,
    STATE_DIM,
)
from so101_dual_dinov2_act.models import build_act_model, kl_divergence


def test_model():
    print("=" * 60)
    print("🧪 SO-101 양안 DINOv2 ACT 모델 입출력 Shape 및 순전파/역전파 검증")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    print(f"백본: {BACKBONE}")
    print(f"카메라: {CAMERA_NAMES} (총 {len(CAMERA_NAMES)}대)")
    print(f"입력 이미지 크기: {IMAGE_SIZE} (H, W)")
    print(f"액션 청크 크기: {CHUNK_SIZE}, 로봇 상태 차원: {STATE_DIM}-DOF")

    # 1. 모델 빌드
    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=CHUNK_SIZE,
        camera_names=CAMERA_NAMES,
        backbone_name=BACKBONE,
        hidden_dim=HIDDEN_DIM,
        train_backbone=False,
    ).to(device)

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[파라미터 통계]")
    print(f"  총 파라미터: {total_params / 1e6:.2f} M")
    print(f"  학습 대상 파라미터 (백본 Freeze 상태): {trainable_params / 1e6:.2f} M")

    # 2. Dummy 입력 생성
    bs = 2
    num_cam = len(CAMERA_NAMES)
    H, W = IMAGE_SIZE

    dummy_qpos = torch.randn(bs, STATE_DIM).to(device)
    dummy_image = torch.randn(bs, num_cam, 3, H, W).to(device)
    dummy_actions = torch.randn(bs, CHUNK_SIZE, STATE_DIM).to(device)
    dummy_is_pad = torch.zeros(bs, CHUNK_SIZE, dtype=torch.bool).to(device)

    print(f"\n[입력 텐서 형상]")
    print(f"  qpos:        {dummy_qpos.shape}")
    print(f"  image:       {dummy_image.shape} (BS, num_cam, 3, H, W)")
    print(f"  actions:     {dummy_actions.shape} (BS, chunk, 6)")
    print(f"  is_pad:      {dummy_is_pad.shape}")

    # 3. 학습 모드 Forward 및 손실 계산 (CVAE 활성화)
    print("\n[테스트 1] 학습 모드 Forward & Backward 검증...")
    model.train()
    a_hat, is_pad_hat, (mu, logvar) = model(
        qpos=dummy_qpos,
        image=dummy_image,
        actions=dummy_actions,
        is_pad=dummy_is_pad,
    )

    print(f"  -> a_hat 예측 형상:       {a_hat.shape} (기대값: [{bs}, {CHUNK_SIZE}, {STATE_DIM}])")
    print(f"  -> is_pad_hat 예측 형상:   {is_pad_hat.shape} (기대값: [{bs}, {CHUNK_SIZE}, 1])")
    print(f"  -> CVAE mu 형상:          {mu.shape} (기대값: [{bs}, 32])")
    print(f"  -> CVAE logvar 형상:      {logvar.shape} (기대값: [{bs}, 32])")

    assert a_hat.shape == (bs, CHUNK_SIZE, STATE_DIM), "액션 예측 Shape 불일치!"
    assert mu.shape == (bs, 32), "CVAE Latent mu Shape 불일치!"

    # 손실 및 역전파 검증
    l1_loss = torch.nn.functional.l1_loss(dummy_actions, a_hat)
    kld, _, _ = kl_divergence(mu, logvar)
    total_loss = l1_loss + 0.1 * kld[0]
    total_loss.backward()
    print(f"  -> Total Loss: {total_loss.item():.4f} (L1: {l1_loss.item():.4f}, KL: {kld[0].item():.4f})")
    print("  -> 역전파(Backward) 기울기 계산 성공! ✅")

    # 4. 추론 모드 Forward 검증 (CVAE 잠재 변수 z=0 샘플링)
    print("\n[테스트 2] 실시간 추론 모드 (actions=None) 검증...")
    model.eval()
    with torch.no_grad():
        a_hat_eval, _, (mu_eval, logvar_eval) = model(
            qpos=dummy_qpos,
            image=dummy_image,
        )
    print(f"  -> 추론 모드 a_hat 형상:   {a_hat_eval.shape} (기대값: [{bs}, {CHUNK_SIZE}, {STATE_DIM}])")
    print(f"  -> 추론 모드 mu:          {mu_eval} (기대값: None)")
    assert a_hat_eval.shape == (bs, CHUNK_SIZE, STATE_DIM)
    assert mu_eval is None
    print("  -> 추론 모드 동작 성공! ✅")

    print("\n" + "=" * 60)
    print("🎉 모든 양안 DINOv2 ACT 모델 단위 테스트 통과 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
