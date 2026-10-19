"""SO-101 픽앤플레이스 ACT 모델 학습 스크립트 (Phase 5).

수집된 32개 HDF5 에피소드 데이터셋을 기반으로
CVAE 인코더와 DETR 트랜스포머 디코더를 학습시킵니다.
"""

import argparse
import csv
import json
import os
import pickle
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 안전하게 저장
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from so101_act.config import (
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
    KL_WEIGHT,
    LATENT_DIM,
    LR,
    LR_BACKBONE,
    NHEADS,
    NUM_EPOCHS,
    SAVE_EVERY,
    SEED,
    STATE_DIM,
    WEIGHT_DECAY,
)
from so101_act.dataset import load_so101_dataloaders
from so101_act.models import build_act_model, kl_divergence


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
    """학습 및 검증 손실 곡선(Total Loss, L1, KL) 그래프 생성 및 저장"""
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
    fig.suptitle("SO-101 ACT Policy Training History", fontsize=15, fontweight="bold")

    # 1. Total Loss
    axes[0].plot(epochs, train_loss, label="Train Total", color="#1f77b4", linewidth=1.5)
    axes[0].plot(epochs, val_loss, label="Val Total", color="#ff7f0e", linewidth=1.8)
    best_idx = int(np.argmin(val_loss))
    axes[0].scatter(
        [epochs[best_idx]],
        [val_loss[best_idx]],
        color="red",
        s=80,
        zorder=5,
        label=f"Best: {val_loss[best_idx]:.4f} (Ep {epochs[best_idx]})",
    )
    axes[0].set_title("Total Loss (L1 + KL)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2. L1 Reconstruction Loss
    axes[1].plot(epochs, train_l1, label="Train L1", color="#2ca02c", linewidth=1.5)
    axes[1].plot(epochs, val_l1, label="Val L1", color="#d62728", linewidth=1.8)
    axes[1].set_title("L1 Action Imitation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L1 Error (rad)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # 3. KL Divergence
    axes[2].plot(epochs, train_kl, label="Train KL", color="#9467bd", linewidth=1.5)
    axes[2].plot(epochs, val_kl, label="Val KL", color="#8c564b", linewidth=1.8)
    axes[2].set_title("CVAE KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_training_summary(history: list, ckpt_dir: str):
    """학습 메트릭을 JSON, CSV, PNG 그래프로 저장"""
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. JSON 저장
    json_path = os.path.join(ckpt_dir, "train_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 2. CSV 저장
    csv_path = os.path.join(ckpt_dir, "train_summary.csv")
    fieldnames = ["epoch", "train_loss", "train_l1", "train_kl", "val_loss", "val_l1", "val_kl", "is_best", "dt"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    # 3. PNG 그래프 생성
    png_path = os.path.join(ckpt_dir, "train_curve.png")
    try:
        plot_training_curves(history, png_path)
    except Exception as e:
        print(f"[경고] 그래프 생성 중 예외 발생: {e}")


def main():
    parser = argparse.ArgumentParser(description="SO-101 ACT 모델 학습 (Phase 5)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"학습 에포크 수 (기본값: {NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"배치 크기 (기본값: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LR, help=f"학습률 (기본값: {LR})")
    parser.add_argument("--lr-backbone", type=float, default=LR_BACKBONE, help=f"비전 백본(ResNet) 학습률 (기본값: {LR_BACKBONE})")
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT, help=f"KL 발산 손실 가중치 (기본값: {KL_WEIGHT})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"청크 크기 (기본값: {CHUNK_SIZE})")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help=f"데이터셋 경로 (기본값: {DATA_DIR})")
    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR, help=f"체크포인트 저장 경로 (기본값: {CKPT_DIR})")
    parser.add_argument("--exp-name", type=str, default=None, help="고유 실험 이름 (지정 시 ckpt/experiments/{exp_name} 에 자동 저장)")
    parser.add_argument("--episodes", type=str, default=None, help="학습에 사용할 에피소드 번호 범위 (예: '56-76' 또는 '0-20,56-76', 기본값: 전체)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help=f"체크포인트 저장 주기 에포크 단위 (기본값: {SAVE_EVERY})")
    parser.add_argument("--seed", type=int, default=SEED, help="난수 시드")
    args = parser.parse_args()

    # 실험 이름 지정 시 고유 폴더로 라우팅
    if args.exp_name:
        args.ckpt_dir = os.path.join(BASE_DIR, "ckpt", "experiments", args.exp_name)

    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 실험 하이퍼파라미터 설정 영구 저장 (A/B 테스트 비교용)
    config_record = {
        "exp_name": args.exp_name or os.path.basename(args.ckpt_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "kl_weight": args.kl_weight,
        "chunk_size": args.chunk_size,
        "episodes": args.episodes or "all",
        "save_every": args.save_every,
        "hidden_dim": HIDDEN_DIM,
        "dim_feedforward": DIM_FEEDFORWARD,
        "enc_layers": ENC_LAYERS,
        "dec_layers": DEC_LAYERS,
        "nheads": NHEADS,
        "latent_dim": LATENT_DIM,
        "backbone": BACKBONE,
        "data_dir": os.path.abspath(args.data_dir),
        "ckpt_dir": os.path.abspath(args.ckpt_dir),
    }
    with open(os.path.join(args.ckpt_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2, ensure_ascii=False)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print(" 🚀 [SO-101 ACT 모델 학습 시작]")
    print("=" * 70)
    print(f" ⚙️ 연산 장치       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" 📂 데이터셋 경로   : {os.path.abspath(args.data_dir)}")
    print(f" 📁 가중치 저장 경로: {os.path.abspath(args.ckpt_dir)}")
    print(f" 🎯 하이퍼파라미터  : Epochs={args.epochs}, Batch={args.batch_size}, Chunk={args.chunk_size}, KL={args.kl_weight}, LR={args.lr}, Episodes={args.episodes or 'all'}")
    print("=" * 70)

    # 1. 데이터셋 로더 구성
    train_loader, val_loader, norm_stats = load_so101_dataloaders(
        data_dir=args.data_dir,
        camera_names=CAMERA_NAMES,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        seed=args.seed,
        episodes=args.episodes,
    )

    # 정규화 통계 저장 (추론 시 필수 재사용)
    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(norm_stats, f)
    print(f"💾 데이터셋 정규화 통계 저장 완료 ➔ {stats_path}")

    # 2. ACT 모델 빌드
    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=args.chunk_size,
        camera_names=CAMERA_NAMES,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
    )
    model.to(device)

    # 파라미터 수 계산
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 모델 파라미터 총합: {n_params / 1e6:.2f} M (백본: {BACKBONE}, 은닉차원: {HIDDEN_DIM})")

    # 3. 옵티마이저 구성 (백본과 트랜스포머의 학습률 분리)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=WEIGHT_DECAY)

    # 4. 학습 루프
    min_val_loss = float("inf")
    best_epoch = 0
    history = []

    print("\n" + "-" * 70)
    print(" 📈 에포크별 학습 진행 상황")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # [검증 단계 (Validation)]
        model.eval()
        val_losses = []
        val_l1s = []
        val_kls = []

        with torch.inference_mode():
            for data in val_loader:
                loss_dict = compute_loss(model, data, args.kl_weight, device)
                val_losses.append(loss_dict["loss"].item())
                val_l1s.append(loss_dict["l1"].item())
                val_kls.append(loss_dict["kl"].item())

        avg_val_loss = np.mean(val_losses)
        avg_val_l1 = np.mean(val_l1s)
        avg_val_kl = np.mean(val_kls)

        # 최고 성능 모델 저장 (policy_best.ckpt)
        is_best = False
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_epoch = epoch
            is_best = True
            best_ckpt_path = os.path.join(args.ckpt_dir, "policy_best.ckpt")
            torch.save(model.state_dict(), best_ckpt_path)

        # [학습 단계 (Training)]
        model.train()
        train_losses = []
        train_l1s = []
        train_kls = []

        for data in train_loader:
            optimizer.zero_grad()
            loss_dict = compute_loss(model, data, args.kl_weight, device)
            loss_dict["loss"].backward()
            optimizer.step()

            train_losses.append(loss_dict["loss"].item())
            train_l1s.append(loss_dict["l1"].item())
            train_kls.append(loss_dict["kl"].item())

        avg_train_loss = np.mean(train_losses)
        avg_train_l1 = np.mean(train_l1s)
        avg_train_kl = np.mean(train_kls)

        dt = time.time() - t0
        best_marker = "🌟 BEST!" if is_best else ""

        # 에포크 기록 누적
        record = {
            "epoch": epoch,
            "train_loss": float(round(avg_train_loss, 5)),
            "train_l1": float(round(avg_train_l1, 5)),
            "train_kl": float(round(avg_train_kl, 5)),
            "val_loss": float(round(avg_val_loss, 5)),
            "val_l1": float(round(avg_val_l1, 5)),
            "val_kl": float(round(avg_val_kl, 5)),
            "is_best": bool(is_best),
            "dt": float(round(dt, 2)),
        }
        history.append(record)

        # 진행 로그 출력 (10 에포크마다 또는 베스트 갱신 시 상세 출력)
        if epoch == 1 or epoch % 10 == 0 or is_best:
            print(
                f"Epoch [{epoch:4d}/{args.epochs}] ({dt:4.1f}s) | "
                f"Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}, KL: {avg_train_kl:.4f}) | "
                f"Val Loss: {avg_val_loss:.4f} (L1: {avg_val_l1:.4f}, KL: {avg_val_kl:.4f}) {best_marker}"
            )

        # 10 에포크마다 중간 로그 디스크 동기화 (비정상 종료 대비)
        if epoch % 10 == 0:
            save_training_summary(history, args.ckpt_dir)

        # 주기적 체크포인트 저장 (save_every 에포크 단위)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            periodic_ckpt_path = os.path.join(args.ckpt_dir, f"policy_epoch_{epoch}.ckpt")
            torch.save(model.state_dict(), periodic_ckpt_path)

    # 최종 체크포인트 저장 (policy_last.ckpt)
    last_ckpt_path = os.path.join(args.ckpt_dir, "policy_last.ckpt")
    torch.save(model.state_dict(), last_ckpt_path)

    # 최종 학습 요약 저장 (JSON, CSV, PNG 그래프)
    save_training_summary(history, args.ckpt_dir)

    # 전체 실험 누적 CSV 기록 (A/B 테스트 비교용)
    try:
        summary_csv_path = os.path.join(BASE_DIR, "ckpt", "all_train_experiments.csv")
        write_header = not os.path.exists(summary_csv_path)
        with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "exp_name", "epochs", "best_epoch", "min_val_loss", "final_val_loss", "final_val_l1", "final_val_kl", "lr", "kl_weight", "chunk_size", "ckpt_dir"])
            writer.writerow([
                config_record["timestamp"],
                config_record["exp_name"],
                args.epochs,
                best_epoch,
                round(float(min_val_loss), 5),
                round(float(history[-1]["val_loss"]), 5),
                round(float(history[-1]["val_l1"]), 5),
                round(float(history[-1]["val_kl"]), 5),
                args.lr,
                args.kl_weight,
                args.chunk_size,
                os.path.abspath(args.ckpt_dir)
            ])
        print(f"📑 [전체 학습 실험 요약 CSV 누적 기록 완료] ➔ {summary_csv_path}")
    except Exception as csv_err:
        print(f"[경고] 전체 실험 CSV 기록 중 예외: {csv_err}")

    print("\n" + "=" * 70)
    print(f"🎉 [학습 완료!] 최저 검증 손실(Val Loss): {min_val_loss:.5f} (Epoch {best_epoch})")
    print(f"📁 최고 모델 가중치: {os.path.join(args.ckpt_dir, 'policy_best.ckpt')}")
    print(f"📊 학습 로그 요약: {os.path.join(args.ckpt_dir, 'train_summary.csv')}")
    print(f"📈 학습 손실 곡선: {os.path.join(args.ckpt_dir, 'train_curve.png')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
