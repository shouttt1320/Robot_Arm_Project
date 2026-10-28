"""SO-101 DAPIER-ACT 학습 스크립트.

양팔 12 DOF + OS30A 3D PointCloud (DP3) + Dual Wrist RGB (ResNet-18)
"""

import argparse
import os
from pathlib import Path
import pickle
import time

import torch
from torch.utils.data import DataLoader, random_split

from .config import DapierACTConfig
from .dataset import DapierACTDataset
from .model import DapierACTMultimodal, dapier_act_loss


def train_one_epoch(model, dataloader, optimizer, device, kl_weight):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0

    for batch in dataloader:
        qpos = batch["qpos"].to(device, non_blocking=True)
        pointcloud = batch["pointcloud"].to(device, non_blocking=True)
        left_wrist = batch["left_wrist"].to(device, non_blocking=True)
        right_wrist = batch["right_wrist"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)

        optimizer.zero_grad()

        pred_actions, mu, logvar = model(
            qpos=qpos,
            pointcloud=pointcloud,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            actions=actions,
            is_pad=is_pad,
        )

        loss_dict = dapier_act_loss(
            pred_actions,
            actions,
            is_pad=is_pad,
            mu=mu,
            logvar=logvar,
            kl_weight=kl_weight,
        )

        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss_dict["loss"].item()
        total_l1 += loss_dict["l1"].item()
        total_kl += loss_dict["kl"].item()

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "l1": total_l1 / num_batches,
        "kl": total_kl / num_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, kl_weight):
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0

    for batch in dataloader:
        qpos = batch["qpos"].to(device, non_blocking=True)
        pointcloud = batch["pointcloud"].to(device, non_blocking=True)
        left_wrist = batch["left_wrist"].to(device, non_blocking=True)
        right_wrist = batch["right_wrist"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)

        pred_actions, mu, logvar = model(
            qpos=qpos,
            pointcloud=pointcloud,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            actions=actions,
            is_pad=is_pad,
        )

        loss_dict = dapier_act_loss(
            pred_actions,
            actions,
            is_pad=is_pad,
            mu=mu,
            logvar=logvar,
            kl_weight=kl_weight,
        )

        total_loss += loss_dict["loss"].item()
        total_l1 += loss_dict["l1"].item()
        total_kl += loss_dict["kl"].item()

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "l1": total_l1 / num_batches,
        "kl": total_kl / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="SO-101 DAPIER-ACT 학습기")
    parser.add_argument("--dataset_dir", type=str, required=True, help="HDF5 에피소드 저장 폴더 경로")
    parser.add_argument("--output_dir", type=str, default="ckpt/so101_dapier_act", help="체크포인트 저장 디렉토리")
    parser.add_argument("--epochs", type=int, default=2000, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="트랜스포머 및 헤드 학습률")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="손목 ResNet 백본 학습률")
    parser.add_argument("--kl_weight", type=float, default=10.0, help="CVAE KL divergence 가중치")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="검증 데이터셋 비율")
    parser.add_argument("--save_interval", type=int, default=100, help="체크포인트 저장 주기(에폭)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DapierACTConfig()
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.lr_backbone = args.lr_backbone
    config.kl_weight = args.kl_weight
    config.epochs = args.epochs

    device = torch.device(args.device)
    print("=" * 70)
    print(" 🚀 [SO-101 DAPIER-ACT 멀티모달 모방학습 시작]")
    print(f" • 데이터셋 경로 : {args.dataset_dir}")
    print(f" • 체크포인트    : {args.output_dir}")
    print(f" • 디바이스      : {device}")
    print(f" • 총 에폭수     : {args.epochs} (Batch Size: {args.batch_size})")
    print("=" * 70)

    # 1. 데이터셋 로드
    full_dataset = DapierACTDataset(args.dataset_dir, config=config)

    # 통계 저장
    stats_save_path = output_dir / "stats.pkl"
    with open(stats_save_path, "wb") as f:
        pickle.dump(full_dataset.stats, f)
    print(f"💾 데이터 정규화 통계 저장 완료: {stats_save_path}")

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda")
    )

    # 2. 모델 생성
    model = DapierACTMultimodal(config=config)
    # 데이터셋에서 산출된 포인트클라우드 min/max를 모델 인코더에 주입
    model.point_encoder.set_point_limits(full_dataset.stats["point_min"], full_dataset.stats["point_max"])
    model.to(device)

    # 3. 옵티마이저 (백본은 1e-5, 트랜스포머는 1e-4)
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr, "weight_decay": config.weight_decay},
            {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": config.weight_decay},
        ]
    )

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.kl_weight)
        val_metrics = evaluate(model, val_loader, device, args.kl_weight)
        ep_duration = time.time() - ep_start

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:04d}/{args.epochs:04d}] ({ep_duration:.1f}s) "
                f"Train Loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f}, KL: {train_metrics['kl']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} (L1: {val_metrics['l1']:.4f}, KL: {val_metrics['kl']:.4f})"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                    "stats": full_dataset.stats,
                },
                best_path,
            )

        if epoch % args.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "config": config,
                    "stats": full_dataset.stats,
                },
                ckpt_path,
            )

    total_time = time.time() - start_time
    print(f"\n🎉 [학습 완료] 소요 시간: {total_time/60:.1f}분, 최저 검증 손실: {best_val_loss:.4f}")
    print(f"💾 최종 베스트 모델 저장 위치: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
