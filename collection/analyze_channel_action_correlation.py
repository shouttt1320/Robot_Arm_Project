"""모터값과 ResNet-18 특정 채널(피처맵)의 상관관계 및 인과 민감도 분석 도구.

1. Gradient Attribution: 모터 액션(예: Base 모터)에 대한 512개 채널별 기울기 크기(Importance Ranking)
2. Spatial Saliency: 공 검출 채널(#429) 및 상위 채널의 공간 감도 히트맵
3. Channel Perturbation (좌우 이동 실험): 429번 채널의 피크를 좌우(X=2~17)로 이동시켰을 때 모델의 베이스 각도 예측치 변화 검증
"""

import os
import json
import pickle
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from so101_act.models.detr_vae import build_act_model
from so101_act.models.backbone import build_backbone
from so101_act.models.transformer import build_transformer


def load_model(ckpt_path: str, config_path: str, device: str = "cpu"):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = build_act_model(
        state_dim=6,
        num_queries=cfg.get("chunk_size", 50),
        camera_names=["top", "wrist"],
        hidden_dim=cfg.get("hidden_dim", 512),
        dim_feedforward=cfg.get("dim_feedforward", 3200),
        enc_layers=cfg.get("enc_layers", 4),
        dec_layers=cfg.get("dec_layers", 7),
        nheads=cfg.get("nheads", 8),
        latent_dim=cfg.get("latent_dim", 32),
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, cfg


def run_analysis(
    ckpt_dir: str,
    hdf5_path: str,
    output_png: str,
    step_idx: int = 5,
    ckpt_name: str = "policy_best.ckpt",
):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    config_path = os.path.join(ckpt_dir, "train_config.json")
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt_path, config_path, device=device)

    # 1. HDF5 샘플 데이터 로드
    with h5py.File(hdf5_path, "r") as f:
        top_img_raw = f["observations/images/top"][step_idx]       # (480, 640, 3) BGR or RGB
        wrist_img_raw = f["observations/images/wrist"][step_idx]
        qpos_raw = f["observations/qpos"][step_idx]               # (6,)

    # RGB 확인 및 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    top_t = transform(top_img_raw).unsqueeze(0).to(device)       # (1, 3, 480, 640)
    wrist_t = transform(wrist_img_raw).unsqueeze(0).to(device)
    images = torch.stack([top_t, wrist_t], dim=1)                 # (1, 2, 3, 480, 640)

    # qpos 정규화
    qpos_norm = (qpos_raw - stats["qpos_mean"]) / stats["qpos_std"]
    qpos_t = torch.tensor(qpos_norm, dtype=torch.float32).unsqueeze(0).to(device)

    # 2. 백본 특징 추출 및 Gradient Hook 등록
    features_top, pos_top = model.backbone(images[:, 0])
    feat_top = features_top[0].clone().detach().requires_grad_(True)  # (1, 512, 15, 20)
    pos_top = pos_top[0]

    features_wrist, pos_wrist = model.backbone(images[:, 1])
    feat_wrist = features_wrist[0].detach()
    pos_wrist = pos_wrist[0]

    # 모델 forward (feat_top을 직접 조작하고 기울기를 얻기 위해 중간 단계부터 실행)
    proj_top = model.input_proj(feat_top)
    proj_wrist = model.input_proj(feat_wrist)
    src = torch.cat([proj_top, proj_wrist], axis=3)
    pos = torch.cat([pos_top, pos_wrist], axis=3)

    proprio_input = model.input_proj_robot_state(qpos_t)
    latent_sample = torch.zeros([1, model.latent_dim], dtype=torch.float32, device=device)
    latent_input = model.latent_out_proj(latent_sample)

    hs = model.transformer(
        src, None, model.query_embed.weight, pos,
        latent_input, proprio_input, model.additional_pos_embed.weight
    )[0]
    a_hat = model.action_head(hs)  # (1, chunk_size, 6)

    # Base 모터 액션 (청크 첫 10스텝의 평균)
    base_action = a_hat[0, :10, 0].mean()
    base_action.backward(retain_graph=True)

    # 3. 512개 채널별 기울기 크기 (Action Sensitivity / Attribution)
    grad = feat_top.grad[0]  # (512, 15, 20)
    channel_importance = torch.norm(grad.reshape(512, -1), dim=1).cpu().numpy()  # (512,)
    sorted_indices = np.argsort(channel_importance)[::-1]

    rank_429 = int(np.where(sorted_indices == 429)[0][0]) + 1
    imp_429 = float(channel_importance[429])

    top1_ch = int(sorted_indices[0])
    top2_ch = int(sorted_indices[1])
    top3_ch = int(sorted_indices[2])

    print(f"=== Channel Importance on Base Motor Action ===")
    print(f"Top-1 Channel: #{top1_ch} (Grad Norm: {channel_importance[top1_ch]:.5f})")
    print(f"Top-2 Channel: #{top2_ch} (Grad Norm: {channel_importance[top2_ch]:.5f})")
    print(f"Top-3 Channel: #{top3_ch} (Grad Norm: {channel_importance[top3_ch]:.5f})")
    print(f"Ball Detector Channel #429: Rank {rank_429} / 512 (Grad Norm: {imp_429:.5f})")
    ratio = channel_importance[top1_ch] / (imp_429 + 1e-8)
    print(f"Top-1 is {ratio:.2f}x more influential on Base motor than Ball Channel #429")

    # 4. Perturbation 실험: 429번 채널의 피크 위치를 X=2부터 X=17까지 이동시켜보기
    # 원본 429번 채널의 활성화 패턴 추출
    orig_f429 = feat_top[0, 429].detach().cpu().numpy()
    peak_y, peak_x = np.unravel_index(np.argmax(orig_f429), orig_f429.shape)
    peak_val = orig_f429[peak_y, peak_x]

    x_test_range = list(range(2, 18))
    pred_base_deg_list = []

    # 실제 unnormalization 파라미터
    base_mean = stats["action_mean"][0]
    base_std = stats["action_std"][0]

    with torch.no_grad():
        for tx in x_test_range:
            # 429 채널만 복사하여 피크 가우시안을 (peak_y, tx)로 이동
            feat_perturbed = feat_top.clone().detach()
            # 429번 채널 초기화 후 가우시안 생성
            new_f429 = torch.zeros((15, 20), device=device)
            for gy in range(15):
                for gx in range(20):
                    dist_sq = (gy - peak_y) ** 2 + (gx - tx) ** 2
                    new_f429[gy, gx] = peak_val * np.exp(-dist_sq / (2 * 1.5 ** 2))
            feat_perturbed[0, 429] = new_f429

            proj_top_p = model.input_proj(feat_perturbed)
            src_p = torch.cat([proj_top_p, proj_wrist], axis=3)
            hs_p = model.transformer(
                src_p, None, model.query_embed.weight, pos,
                latent_input, proprio_input, model.additional_pos_embed.weight
            )[0]
            a_p = model.action_head(hs_p)
            norm_action = a_p[0, :5, 0].mean().item()
            rad_action = norm_action * base_std + base_mean
            deg_action = rad_action * (180.0 / np.pi)
            pred_base_deg_list.append(deg_action)

    # 5. 시각화 피규어 생성 (2x2 그리드)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="white")
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # (1) [좌상단] 512개 채널의 Base 모터 영향도 랭킹 막대그래프
    ax0 = axes[0, 0]
    ranks_to_plot = sorted_indices[:30]
    bars = ax0.bar(range(30), channel_importance[ranks_to_plot], color="#4C72B0", alpha=0.85, width=0.7)
    # 429번 채널이 상위 30위 안에 없으면 우측 끝에 별도 표시
    highlight_text = f"Channel #429 (Ball)\nRank: {rank_429} / 512\nGrad: {imp_429:.5f}"
    if 429 in ranks_to_plot:
        idx_in_plot = list(ranks_to_plot).index(429)
        bars[idx_in_plot].set_color("#E74C3C")
    else:
        ax0.axhline(imp_429, color="#E74C3C", linestyle="--", linewidth=2, label=f"Ch #429 Level ({imp_429:.5f})")
        ax0.legend(loc="upper right", fontsize=11)

    ax0.set_xticks(range(30))
    ax0.set_xticklabels([f"#{c}" for c in ranks_to_plot], rotation=60, fontsize=9)
    ax0.set_title(f"1. Top 30 Channels Influencing Base Motor (Gradient Importance)\n[Ball Ch #429 is Rank {rank_429}/512]", fontsize=12, fontweight="bold")
    ax0.set_ylabel("Gradient Norm ||d(Action_Base) / d(Channel)||", fontsize=10)
    ax0.grid(axis="y", linestyle=":", alpha=0.5)

    # (2) [우상단] 원본 영상 위에 공 채널(#429)의 활성화 및 Saliency 표시
    ax1 = axes[0, 1]
    ax1.imshow(top_img_raw)
    # 429번 채널 활성화 맵을 원본 크기로 리사이즈하여 오버레이
    heat_429 = F.interpolate(torch.tensor(orig_f429).unsqueeze(0).unsqueeze(0), size=(480, 640), mode="bilinear", align_corners=False)[0, 0].numpy()
    heat_norm = (heat_429 - heat_429.min()) / (heat_429.max() - heat_429.min() + 1e-8)
    ax1.imshow(heat_norm, cmap="jet", alpha=0.45)
    circle = patches.Circle((peak_x * 32 + 16, peak_y * 32 + 16), 28, edgecolor="red", facecolor="none", linewidth=3)
    ax1.add_patch(circle)
    ax1.set_title(f"2. Ball Detector Channel #429 Feature Map Overlay\n[Peak at grid (x={peak_x}, y={peak_y}) | Activation={peak_val:.2f}]", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # (3) [좌하단] 1위 채널 (#239)의 피처맵 오버레이 (무엇을 보길래 1위인가?)
    ax2 = axes[1, 0]
    top1_feat = feat_top[0, top1_ch].detach().cpu().numpy()
    heat_top1 = F.interpolate(torch.tensor(top1_feat).unsqueeze(0).unsqueeze(0), size=(480, 640), mode="bilinear", align_corners=False)[0, 0].numpy()
    heat_top1_norm = (heat_top1 - heat_top1.min()) / (heat_top1.max() - heat_top1.min() + 1e-8)
    ax2.imshow(top_img_raw)
    ax2.imshow(heat_top1_norm, cmap="inferno", alpha=0.5)
    ax2.set_title(f"3. Rank #1 Channel (#{top1_ch}) Feature Map Overlay\n[Robot Arm / Background Edge - Influences Base Motor {ratio:.1f}x More]", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # (4) [우하단] Perturbation Sensitivity: 429 채널 피크 X 이동에 따른 Base 예측 각도 반응 곡선
    ax3 = axes[1, 1]
    ax3.plot(x_test_range, pred_base_deg_list, marker="o", color="#E74C3C", linewidth=2.5, markersize=7, label="Predicted Base Angle (deg)")
    ax3.axvline(peak_x, color="blue", linestyle=":", linewidth=2, label=f"Real Ball Grid X={peak_x}")
    ax3.set_title("4. Causal Sensitivity: Base Angle vs Channel #429 Peak Shift (X=2~17)\n[Ideal: Linear Tracking | Actual: Flat/Weak Response]", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Channel #429 Injected Peak X Coordinate (0~19 grid)", fontsize=11)
    ax3.set_ylabel("Predicted Base Motor Angle (°)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(fontsize=11)

    # 변화폭(Range) 표기
    delta_deg = max(pred_base_deg_list) - min(pred_base_deg_list)
    ax3.text(
        0.05, 0.15,
        f"Angle Variation Range: {delta_deg:.2f}°\n(If < 2°, model is ignoring ball position!)",
        transform=ax3.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor="#FFEEBA")
    )

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Analysis saved to: {output_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="ckpt/experiments/reach_lr3e5_bk1e5")
    parser.add_argument("--ckpt-name", type=str, default="policy_best.ckpt")
    parser.add_argument("--hdf5", type=str, default="data/so101_pick_place/episode_56.hdf5")
    parser.add_argument("--output", type=str, default="logs/channel_action_correlation.png")
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_analysis(args.ckpt_dir, args.hdf5, args.output, step_idx=args.step, ckpt_name=args.ckpt_name)
