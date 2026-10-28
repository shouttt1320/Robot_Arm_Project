"""SO-101 SmolVLA 체크포인트별 평가 손실(Loss) 측정 및 학습 곡선 시각화 스크립트."""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.rename_processor import rename_batch_keys


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [SmolVLA 학습 곡선 평가] 디바이스: {device}")

    dataset_path = "data/lerobot_so101_pick_place"
    ckpt_root = "ckpt/smolvla_so101/checkpoints"
    steps = ["005000", "010000", "015000", "020000"]
    step_ints = [5000, 10000, 15000, 20000]

    # 1. 공통 데이터셋 및 델타 타임스탬프 설정
    first_ckpt = os.path.join(ckpt_root, steps[0], "pretrained_model")
    first_policy = SmolVLAPolicy.from_pretrained(first_ckpt)
    meta = LeRobotDataset("local/so101_pick_place", root=dataset_path).meta
    rename_map = {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    delta_timestamps = resolve_delta_timestamps(first_policy.config, meta, rename_map)
    del first_policy
    torch.cuda.empty_cache()

    dataset = LeRobotDataset("local/so101_pick_place", root=dataset_path, delta_timestamps=delta_timestamps)
    print(f"📊 데이터셋 로드 완료: 총 {len(dataset)} 프레임")

    # 2. 고정된 시드로 평가용 고정 배치 25개(총 200 샘플) 미리 추출
    torch.manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    eval_batches = []
    print("⏳ 평가용 고정 배치 데이터 수집 중...")
    for idx, batch in enumerate(loader):
        if idx >= 25:  # 25 batches * 8 = 200 samples
            break
        # uint8 -> float32 변환
        for cam_key in ["observation.images.top", "observation.images.wrist"]:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = rename_batch_keys(batch, rename_map)
        eval_batches.append(batch)
    print(f"✅ 평가 배치 {len(eval_batches)}개 (200 샘플) 준비 완료.")

    # 3. 각 체크포인트별 평가 수행
    results = {}
    for step_str, step_num in zip(steps, step_ints):
        ckpt_path = os.path.join(ckpt_root, step_str, "pretrained_model")
        print(f"\n🔍 [Step {step_num:,}] 체크포인트 평가 중: {ckpt_path}")
        t0 = time.time()

        policy = SmolVLAPolicy.from_pretrained(ckpt_path).to(device)
        policy.eval()

        preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=ckpt_path)

        losses = []
        with torch.no_grad():
            for batch in eval_batches:
                processed_batch = preprocessor(batch)
                loss, _ = policy(processed_batch)
                losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses))
        elapsed = time.time() - t0
        results[step_num] = {
            "mean": mean_loss,
            "std": std_loss,
            "losses": losses,
        }
        print(f"   ➔ Mean Loss: {mean_loss:.4f} (±{std_loss:.4f}) | 소요시간: {elapsed:.1f}s")

        del policy
        torch.cuda.empty_cache()

    # 4. 시각화 그래프 생성
    print("\n📈 [그래프 생성] 학습 손실 곡선 플로팅 중...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    x = step_ints
    y = [results[s]["mean"] for s in x]
    y_err = [results[s]["std"] / np.sqrt(len(eval_batches)) for s in x]  # 표준 오차

    # 라인 및 영역
    ax.plot(x, y, marker="o", color="#2563EB", linewidth=2.8, markersize=8, label="SmolVLA Flow Matching Loss", zorder=3)
    ax.fill_between(
        x,
        [val - err for val, err in zip(y, y_err)],
        [val + err for val, err in zip(y, y_err)],
        color="#3B82F6",
        alpha=0.18,
        label="Standard Error (SE)",
    )

    # 각 포인트에 수치 어노테이션
    for s, loss_val in zip(x, y):
        ax.annotate(
            f"{loss_val:.4f}",
            (s, loss_val),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="#1E3A8A",
            bbox=dict(boxstyle="round,pad=0.25", fc="#EFF6FF", ec="#93C5FD", lw=1.2),
        )

    # 초기값 대비 감소율 표시
    reduction_pct = ((y[0] - y[-1]) / y[0]) * 100
    ax.text(
        0.03,
        0.10,
        f"Initial Loss (5k): {y[0]:.4f}\nFinal Loss (20k): {y[-1]:.4f}\nOverall Reduction: -{reduction_pct:.1f}%",
        transform=ax.transAxes,
        fontsize=10.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F8FAFC", ec="#CBD5E1", lw=1.2),
    )

    ax.set_title("SmolVLA Fine-tuning Loss Curve (SO-101 Pick-and-Place)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Flow Matching Loss (MSE)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in x])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="upper right")

    plt.tight_layout()

    out_file1 = "ckpt/smolvla_so101/smolvla_training_curve.png"
    out_file2 = "/home/jjj/.gemini/antigravity-ide/brain/219b695b-6087-4d08-af81-9a4a2f4473b3/smolvla_training_curve.png"
    plt.savefig(out_file1, dpi=300)
    plt.savefig(out_file2, dpi=300)
    plt.close()

    print(f"✅ 그래프 저장 완료:\n  - {out_file1}\n  - {out_file2}")

    # 요약 출력
    print("\n" + "=" * 55)
    print(" 📊 [SmolVLA 체크포인트별 학습 손실 요약]")
    print("=" * 55)
    print(f" {'Step':^10} | {'Mean Loss':^14} | {'Std Dev':^12} | {'비고':^10}")
    print("-" * 55)
    for s in x:
        status = "Start (5k)" if s == 5000 else ("Final (20k)" if s == 20000 else "")
        print(f" {s:^10,d} | {results[s]['mean']:^14.4f} | {results[s]['std']:^12.4f} | {status:^10}")
    print("=" * 55)


if __name__ == "__main__":
    main()
