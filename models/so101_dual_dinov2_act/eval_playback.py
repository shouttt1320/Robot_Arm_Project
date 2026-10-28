"""양안(Dual Eyes) DINOv2 ACT 모델 오프라인 궤적 검증 및 분석 도구.

실제 로봇 구동 전, 저장된 검증 에피소드(Val HDF5)를 모델에 넣어:
1. 실제 사람 정답 궤적 (Ground Truth) vs 모델 예측 궤적 (Predicted Action) 비교.
2. 6개 관절(Base, Shoulder, Elbow, Wrist Pitch, Wrist Roll, Gripper)별 시간 추종 그래프 저장.
3. 평균 L1 절대 오차(MAE) 산출.
"""

import argparse
import os
import pickle
import sys
import time
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

# 프로젝트 루트 경로
cur = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(cur))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.so101_dual_dinov2_act.config import (
    BACKBONE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    HIDDEN_DIM,
    IMAGE_SIZE,
    LATENT_DIM,
    NHEADS,
    STATE_DIM,
    TEMPORAL_AGG_M,
)
from models.so101_dual_dinov2_act.models import build_act_model

JOINT_NAMES = [
    "Joint 1 (Base)",
    "Joint 2 (Shoulder)",
    "Joint 3 (Elbow)",
    "Joint 4 (Wrist Pitch)",
    "Joint 5 (Wrist Roll)",
    "Joint 6 (Gripper)",
]


def eval_offline_playback(
    hdf5_path: str,
    ckpt_path: str = "ckpt/so101_dual_dinov2_act/policy_best.pth",
    stats_path: str = "ckpt/so101_dual_dinov2_act/dataset_stats.pkl",
    output_png: str = "ckpt/so101_dual_dinov2_act/eval_playback.png",
):
    print(f"=== 양안 DINOv2 ACT 오프라인 궤적 분석 시작 ===")
    print(f"  에피소드: {hdf5_path}")
    print(f"  가중치: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 통계치 로드
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    qpos_mean = torch.from_numpy(stats["qpos_mean"]).float().to(device)
    qpos_std = torch.from_numpy(stats["qpos_std"]).float().to(device)
    action_mean = stats["action_mean"]
    action_std = stats["action_std"]

    # 2. 모델 로드
    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=CHUNK_SIZE,
        camera_names=CAMERA_NAMES,
        backbone_name=BACKBONE,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
        train_backbone=False,
    )
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. 전처리 트랜스폼
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. HDF5 데이터 읽기
    with h5py.File(hdf5_path, "r") as root:
        gt_actions = root["action"][()]  # (T, 6)
        qpos_all = root["observations/qpos"][()]  # (T, 6)
        T = gt_actions.shape[0]

        # 이미지 로드
        images_dict = {}
        for cam_name in CAMERA_NAMES:
            if f"observations/images/{cam_name}" in root:
                images_dict[cam_name] = root[f"observations/images/{cam_name}"][()]
            else:
                raise KeyError(f"카메라 {cam_name} 없음")

    # 5. 오프라인 추론 시뮬레이션 (Temporal Ensembling)
    all_pred_actions = np.zeros((T, STATE_DIM))
    all_time_actions = np.zeros((T, T + CHUNK_SIZE, STATE_DIM))

    weights = np.exp(-TEMPORAL_AGG_M * np.arange(CHUNK_SIZE))

    print(f"총 {T} 스텝 추론 시뮬레이션 진행 중...")
    with torch.no_grad():
        for t in range(T):
            # Qpos 정규화
            qpos_t = torch.from_numpy(qpos_all[t]).float().to(device)
            qpos_norm = ((qpos_t - qpos_mean) / qpos_std).unsqueeze(0)

            # 이미지 전처리
            cam_tensors = []
            for cam_name in CAMERA_NAMES:
                img_np = images_dict[cam_name][t]
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                img_t = transform(img_t)
                cam_tensors.append(img_t)
            image_input = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(device)

            # 모델 순전파
            a_hat, _, _ = model(qpos=qpos_norm, image=image_input)
            a_hat_unnorm = a_hat[0].cpu().numpy() * action_std + action_mean

            all_time_actions[t, t:t + CHUNK_SIZE] = a_hat_unnorm

            # Temporal Aggregation 계산
            actions_for_curr_step = all_time_actions[:t + 1, t]
            active_weights = weights[:t + 1][::-1]
            active_weights = active_weights / active_weights.sum()
            all_pred_actions[t] = np.sum(actions_for_curr_step * active_weights[:, None], axis=0)

    # 6. 관절별 궤적 비교 그래프 렌더링
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    timesteps = np.arange(T)
    total_l1_errors = []

    for j in range(STATE_DIM):
        gt = gt_actions[:, j]
        pred = all_pred_actions[:, j]
        mae = np.mean(np.abs(gt - pred))
        total_l1_errors.append(mae)

        axes[j].plot(timesteps, gt, label="Ground Truth (사람)", color="black", linewidth=2.0)
        axes[j].plot(timesteps, pred, label="DINOv2 ACT 예측", color="crimson", linestyle="--", linewidth=1.8)
        axes[j].set_title(f"{JOINT_NAMES[j]} (MAE: {mae:.4f} rad)", fontsize=12, fontweight="bold")
        axes[j].set_xlabel("Time step")
        axes[j].set_ylabel("Angle (rad)")
        axes[j].legend(loc="upper right")
        axes[j].grid(True, alpha=0.3)

    plt.suptitle(f"DINOv2 ACT 오프라인 궤적 추종 검증 ({os.path.basename(hdf5_path)})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

    mean_mae = np.mean(total_l1_errors)
    print(f"\n✅ 오프라인 궤적 분석 완료!")
    print(f"  • 전 관절 평균 MAE: {mean_mae:.4f} rad")
    print(f"  • 분석 그래프 저장 위치: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="양안 DINOv2 ACT 오프라인 궤적 검증")
    parser.add_argument("--hdf5", type=str, default="data/so101_dual_eyes/episode_0.hdf5")
    parser.add_argument("--ckpt", type=str, default="ckpt/so101_dual_dinov2_act/policy_best.pth")
    parser.add_argument("--stats", type=str, default="ckpt/so101_dual_dinov2_act/dataset_stats.pkl")
    parser.add_argument("--output", type=str, default="ckpt/so101_dual_dinov2_act/eval_playback.png")
    args = parser.parse_args()
    eval_offline_playback(args.hdf5, args.ckpt, args.stats, args.output)
