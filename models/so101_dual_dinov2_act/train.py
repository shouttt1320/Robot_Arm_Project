"""양안(Dual Eyes) 카메라 DINOv2 ACT 모델 학습 스크립트."""

import argparse
import csv
import json
import os
import pickle
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from so101_dual_dinov2_act.config import (
    BACKBONE,
    BASE_DIR,
    BATCH_SIZE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
    DATA_DIR,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    HIDDEN_DIM,
    IMAGE_SIZE,
    KL_WEIGHT,
    LATENT_DIM,
    LR,
    LR_BACKBONE,
    NHEADS,
    NUM_EPOCHS,
    SAVE_EVERY,
    SEED,
    STATE_DIM,
    TRAIN_BACKBONE,
    WEIGHT_DECAY,
)
from so101_dual_dinov2_act.dataset import load_dual_eyes_dataloaders
from so101_dual_dinov2_act.models import build_act_model, kl_divergence


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(model, data, kl_weight: float, device: torch.device):
    """L1 모방 오차 및 CVAE KL 발산 손실 계산"""
    image_data, qpos_data, action_data, is_pad_data = data

    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad_data = is_pad_data.to(device)

    # Forward Pass
    a_hat, is_pad_hat, (mu, logvar) = model(
        qpos=qpos_data,
        image=image_data,
        actions=action_data,
        is_pad=is_pad_data,
    )

    # 1. L1 액션 모방 손실 (패딩 구간 마스킹 적용)
    all_l1 = F.l1_loss(action_data, a_hat, reduction="none")
    l1_loss = (all_l1 * ~is_pad_data.unsqueeze(-1)).mean()

    # 2. CVAE KL 발산 손실
    total_kld, _, _ = kl_divergence(mu, logvar)
    kl_loss = total_kld[0]

    # 3. 종합 손실
    total_loss = l1_loss + kl_weight * kl_loss

    return {
        "loss": total_loss,
        "l1": l1_loss,
        "kl": kl_loss,
    }


def plot_training_curves(history: list, save_path: str):
    """학습 및 검증 손실 곡선 그래프 저장"""
    if not history:
        return

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_l1 = [r["train_l1"] for r in history]
    val_l1 = [r["val_l1"] for r in history]
    train_kl = [r["train_kl"] for r in history]
    val_kl = [r["val_kl"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", color="royalblue")
    axes[0].plot(epochs, val_loss, label="Val Loss", color="crimson", linestyle="--")
    axes[0].set_title("Total Loss (L1 + KL)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_l1, label="Train L1", color="royalblue")
    axes[1].plot(epochs, val_l1, label="Val L1", color="crimson", linestyle="--")
    axes[1].set_title("L1 Action Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L1 Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, train_kl, label="Train KL", color="royalblue")
    axes[2].plot(epochs, val_kl, label="Val KL", color="crimson", linestyle="--")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_dual_eyes_act(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SO-101 양안 DINOv2 ACT 모델 학습 시작 (Device: {device}) ===")
    print(f"  백본 모델: {args.backbone} (Train Backbone: {args.train_backbone})")
    print(f"  카메라 목록: {args.camera_names}")
    print(f"  입력 해상도: {args.image_size} (14의 배수)")
    print(f"  배치 크기: {args.batch_size}, 에포크: {args.num_epochs}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 1. 데이터셋 로드
    train_loader, val_loader, norm_stats = load_dual_eyes_dataloaders(
        data_dir=args.data_dir,
        camera_names=args.camera_names,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
    )

    # 정규화 통계치 저장
    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(norm_stats, f)
    print(f"[Stats] 정규화 통계치 저장 완료: {stats_path}")

    # 2. 모델 인스턴스 생성
    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=args.chunk_size,
        camera_names=args.camera_names,
        backbone_name=args.backbone,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
        train_backbone=args.train_backbone,
    ).to(device)

    # 파라미터 그룹 분리 (백본 미세조정 여부에 따른 LR 분리)
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    history = []
    log_csv_path = os.path.join(args.ckpt_dir, "train_log.csv")

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_l1", "train_kl", "val_loss", "val_l1", "val_kl", "time_s"])

    start_time = time.time()

    # 3. 학습 루프
    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        train_l1s = []
        train_kls = []

        for data in train_loader:
            optimizer.zero_grad()
            loss_dict = compute_loss(model, data, args.kl_weight, device)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss_dict["loss"].item())
            train_l1s.append(loss_dict["l1"].item())
            train_kls.append(loss_dict["kl"].item())

        # 검증 루프
        model.eval()
        val_losses = []
        val_l1s = []
        val_kls = []
        with torch.no_grad():
            for data in val_loader:
                loss_dict = compute_loss(model, data, args.kl_weight, device)
                val_losses.append(loss_dict["loss"].item())
                val_l1s.append(loss_dict["l1"].item())
                val_kls.append(loss_dict["kl"].item())

        t_loss = np.mean(train_losses)
        t_l1 = np.mean(train_l1s)
        t_kl = np.mean(train_kls)
        v_loss = np.mean(val_losses)
        v_l1 = np.mean(val_l1s)
        v_kl = np.mean(val_kls)
        epoch_elapsed = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": t_loss,
            "train_l1": t_l1,
            "train_kl": t_kl,
            "val_loss": v_loss,
            "val_l1": v_l1,
            "val_kl": v_kl,
        })

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, t_loss, t_l1, t_kl, v_loss, v_l1, v_kl, f"{epoch_elapsed:.2f}"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}/{args.num_epochs}] "
                  f"Train: {t_loss:.4f} (L1: {t_l1:.4f}, KL: {t_kl:.4f}) | "
                  f"Val: {v_loss:.4f} (L1: {v_l1:.4f}, KL: {v_kl:.4f}) | "
                  f"{epoch_elapsed:.1f}s")

        # 최고 성능 모델 체크포인트 저장
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": v_loss},
                os.path.join(args.ckpt_dir, "policy_best.pth"),
            )

        # 주기별 체크포인트 및 시각화 저장
        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": v_loss},
                os.path.join(args.ckpt_dir, f"policy_epoch_{epoch}.pth"),
            )
            plot_training_curves(history, os.path.join(args.ckpt_dir, "training_curves.png"))

    total_time = time.time() - start_time
    print(f"=== 학습 완료 (총 소요 시간: {total_time/60:.1f}분) ===")
    print(f"최고 Val Loss: {best_val_loss:.4f} -> 저장 위치: {os.path.join(args.ckpt_dir, 'policy_best.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="양안 DINOv2 ACT 모델 학습")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--train_backbone", action="store_true", default=TRAIN_BACKBONE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--kl_weight", type=float, default=KL_WEIGHT)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--camera_names", nargs="+", default=CAMERA_NAMES)
    parser.add_argument("--image_size", nargs=2, type=int, default=IMAGE_SIZE)

    args = parser.parse_args()
    args.image_size = tuple(args.image_size)
    train_dual_eyes_act(args)
