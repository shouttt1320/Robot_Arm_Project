"""ACT 모델 오프라인 재생 검증 도구 (방법 1: Offline Sanity Check).

실제 로봇 하드웨어 연결 없이, 저장된 에피소드(episode_0.hdf5)의 카메라 영상과 qpos를
추론 파이프라인에 그대로 흘려보내어:
  1. 실제 사람 정답 궤적 (Ground Truth Action)
  2. 모델 예측 궤적 (Temporal Aggregation 적용)
  3. 모델 예측 궤적 (No Aggregation, 즉각 0번 액션)
을 6개 관절별로 1:1 비교 플롯팅하고, 모델이 앞으로 나아가는지/제자리에 굳는지 판정합니다.
"""

import argparse
import os
import pickle
import sys
import time
import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

# 프로젝트 루트 sys.path 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from so101_act.config import (
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    HIDDEN_DIM,
    LATENT_DIM,
    NHEADS,
    STATE_DIM,
    TEMPORAL_AGG_M,
)
from so101_act.models import build_act_model

JOINT_NAMES = [
    "Joint 1 (Base)",
    "Joint 2 (Shoulder)",
    "Joint 3 (Elbow)",
    "Joint 4 (Wrist Pitch)",
    "Joint 5 (Wrist Roll)",
    "Joint 6 (Gripper)",
]


def run_offline_playback(
    hdf5_path: str = "data/so101_pick_place/episode_0.hdf5",
    ckpt_path: str = None,
    stats_path: str = None,
    output_png: str = "offline_playback_eval.png",
):
    ckpt_path = ckpt_path or os.path.join(CKPT_DIR, "policy_best.ckpt")
    stats_path = stats_path or os.path.join(os.path.dirname(ckpt_path), "dataset_stats.pkl")

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 데이터가 없습니다: {hdf5_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"통계 파일이 없습니다: {stats_path}")

    print("\n" + "=" * 70)
    print(" 🧪 [방법 1: ACT 모델 오프라인 재생 검증 (Sanity Check)]")
    print(f" • 테스트 데이터셋 : {hdf5_path}")
    print(f" • 모델 가중치     : {ckpt_path}")
    print(f" • 청크 크기       : {CHUNK_SIZE} 스텝")
    print("=" * 70 + "\n")

    # 1. HDF5 데이터 로드
    with h5py.File(hdf5_path, "r") as root:
        gt_actions = root["action"][()]          # (T, 6)
        qpos_all = root["observations/qpos"][()] # (T, 6)
        top_images = root["observations/images/top"][()]     # (T, 480, 640, 3)
        wrist_images = root["observations/images/wrist"][()] # (T, 480, 640, 3)
        T = len(gt_actions)

    print(f"✅ 에피소드 데이터 로드 완료: 총 {T} 프레임")

    # 2. 정규화 통계 로드
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qpos_mean = torch.from_numpy(stats["qpos_mean"]).float().to(device)
    qpos_std = torch.from_numpy(stats["qpos_std"]).float().to(device)
    action_mean = stats["action_mean"]
    action_std = stats["action_std"]

    # 3. 모델 빌드
    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=CHUNK_SIZE,
        camera_names=CAMERA_NAMES,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 4. Temporal Aggregation 버퍼 초기화
    total_time = T + CHUNK_SIZE
    all_time_actions = torch.zeros([total_time, total_time, STATE_DIM])
    weights = np.exp(-TEMPORAL_AGG_M * np.arange(CHUNK_SIZE))
    weights = torch.from_numpy(weights).float()

    pred_agg_list = []
    pred_noagg_list = []

    print(f"⚙️  {T} 프레임에 대해 오프라인 추론 시뮬레이션 중 (장치: {device})...")
    t_start = time.time()

    with torch.inference_mode():
        for t in range(T):
            # qpos 전처리
            curr_qpos = qpos_all[t]
            qpos_t = torch.from_numpy(curr_qpos).float().to(device)
            qpos_norm = (qpos_t - qpos_mean) / qpos_std
            qpos_input = qpos_norm.unsqueeze(0)

            # 이미지 전처리 (HDF5의 RGB 프레임)
            top_t = torch.from_numpy(top_images[t]).permute(2, 0, 1).float() / 255.0
            wrist_t = torch.from_numpy(wrist_images[t]).permute(2, 0, 1).float() / 255.0
            top_t = normalize_img(top_t)
            wrist_t = normalize_img(wrist_t)
            img_input = torch.stack([top_t, wrist_t], dim=0).unsqueeze(0).to(device)

            # ACT 추론 (Prior z=0)
            a_hat, _, _ = model(qpos=qpos_input, image=img_input)
            a_hat_cpu = a_hat.squeeze(0).cpu().numpy()
            action_chunk = a_hat_cpu * action_std + action_mean  # (CHUNK_SIZE, 6)

            # A. No Aggregation (즉각 0번 액션)
            pred_noagg_list.append(action_chunk[0].copy())

            # B. Temporal Aggregation (배포 코드와 100% 동일한 로직)
            action_chunk_t = torch.from_numpy(action_chunk)
            all_time_actions[[t], t : t + CHUNK_SIZE] = action_chunk_t
            actions_for_curr_step = all_time_actions[:, t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]

            k = len(actions_for_curr_step)
            step_weights = weights[:k]
            step_weights = step_weights / step_weights.sum()

            target_agg = (actions_for_curr_step * step_weights.unsqueeze(-1)).sum(axis=0).numpy()
            pred_agg_list.append(target_agg)

            if (t + 1) % 50 == 0 or (t + 1) == T:
                sys.stdout.write(f"\r  진행률: [{t+1:3d}/{T}] (소요: {time.time()-t_start:.1f}s)")
                sys.stdout.flush()

    print("\n✅ 추론 시뮬레이션 완료!")

    pred_agg = np.array(pred_agg_list)       # (T, 6)
    pred_noagg = np.array(pred_noagg_list)   # (T, 6)

    # 5. 오차 분석
    err_agg = np.abs(gt_actions - pred_agg)
    err_noagg = np.abs(gt_actions - pred_noagg)

    mean_err_agg_deg = np.rad2deg(err_agg.mean(axis=0))
    mean_err_noagg_deg = np.rad2deg(err_noagg.mean(axis=0))

    print("\n" + "-" * 70)
    print(f"{'관절 명칭':<22} | {'정답 동작 범위':<14} | {'Agg 평균 오차':<14} | {'NoAgg 평균 오차'}")
    print("-" * 70)
    for j in range(6):
        r_gt = np.rad2deg(gt_actions[:, j].max() - gt_actions[:, j].min())
        print(
            f"{JOINT_NAMES[j]:<22} | {r_gt:6.1f}°        | {mean_err_agg_deg[j]:6.2f}°        | {mean_err_noagg_deg[j]:6.2f}°"
        )
    print("-" * 70)

    # 6. 고해상도 궤적 비교 그래프 저장 (2x3 서브플롯)
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
    axes = axes.flatten()
    steps = np.arange(T)

    for j in range(6):
        ax = axes[j]
        # Ground Truth (실제 사람 정답)
        ax.plot(steps, np.rad2deg(gt_actions[:, j]), label="Ground Truth (Human)", color="black", linewidth=2.2, linestyle="--")
        # Model Pred (Temporal Aggregation)
        ax.plot(steps, np.rad2deg(pred_agg[:, j]), label="Model (with Temporal Agg)", color="#1f77b4", linewidth=2.0)
        # Model Pred (No Aggregation)
        ax.plot(steps, np.rad2deg(pred_noagg[:, j]), label="Model (No Agg / Instant)", color="#ff7f0e", linewidth=1.5, alpha=0.8)

        ax.set_title(f"{JOINT_NAMES[j]} Trajectory (GT vs Pred)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Angle (Degree)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if j == 0:
            ax.legend(loc="upper right", frameon=True)

    axes[4].set_xlabel("Time Step (30Hz)", fontsize=11)
    axes[5].set_xlabel("Time Step (30Hz)", fontsize=11)

    fig.suptitle(
        f"ACT Offline Playback Sanity Check: Model Predictions vs Ground Truth [File: {os.path.basename(hdf5_path)}]",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=160)
    plt.close(fig)

    print(f"\n📊 궤적 비교 그래프 저장 완료 ➔ {output_png}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ACT 모델 오프라인 재생 검증 스크립트")
    parser.add_argument("--episode", type=int, default=0, help="검증할 에피소드 번호")
    parser.add_argument("--file", type=str, default=None, help="HDF5 파일 경로")
    parser.add_argument("--ckpt", type=str, default=None, help="모델 체크포인트")
    parser.add_argument("--output", type=str, default="offline_playback_eval.png", help="저장할 결과 그래프 파일명")
    args = parser.parse_args()

    target_file = args.file or f"data/so101_pick_place/episode_{args.episode}.hdf5"
    run_offline_playback(
        hdf5_path=target_file,
        ckpt_path=args.ckpt,
        output_png=args.output,
    )


if __name__ == "__main__":
    main()
