"""train_multitask.log 로그 파일을 실시간 파싱하여 Train Loss vs Validation Loss 곡선을 시각화하는 도구."""

import os
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_and_plot(log_path: str = "train_multitask.log", save_path: str = "ckpt/smolvla_two_balls/train_val_curve.png"):
    if not os.path.exists(log_path):
        print(f"❌ 로그 파일('{log_path}')이 없습니다.")
        return

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    # 정규식 패턴 (step: 200 또는 step: 1K, eval: step 500: eval_loss=0.1234)
    train_pattern = re.compile(r"step:\s*([0-9.]+[kKmM]?)\s+.*?loss:\s*([0-9.]+)", re.IGNORECASE)
    eval_pattern = re.compile(r"step\s*([0-9.]+[kKmM]?)\s*:\s*eval_loss[=:\s]+([0-9.]+)", re.IGNORECASE)

    def parse_step(s: str) -> int:
        s = s.strip()
        if s.lower().endswith("k"):
            return int(float(s[:-1]) * 1000)
        elif s.lower().endswith("m"):
            return int(float(s[:-1]) * 1000000)
        return int(float(s))

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_eval = eval_pattern.search(line)
            if m_eval:
                step = parse_step(m_eval.group(1))
                loss = float(m_eval.group(2))
                val_steps.append(step)
                val_losses.append(loss)
                continue

            m_train = train_pattern.search(line)
            if m_train and "eval" not in line.lower():
                step = parse_step(m_train.group(1))
                loss = float(m_train.group(2))
                train_steps.append(step)
                train_losses.append(loss)

    if not train_steps and not val_steps:
        print("⚠️ 아직 파싱할 로그 데이터가 충분하지 않습니다.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    if train_steps:
        ax.plot(train_steps, train_losses, color="#2563EB", alpha=0.45, label="Train Loss (Raw)", linewidth=1.2)
        # 이동 평균 (Smooth)
        if len(train_losses) >= 5:
            window = min(15, len(train_losses) // 3)
            smooth_train = np.convolve(train_losses, np.ones(window) / window, mode="valid")
            smooth_steps = train_steps[window - 1 :]
            ax.plot(smooth_steps, smooth_train, color="#1D4ED8", label=f"Train Loss (SMA-{window})", linewidth=2.4)

    if val_steps:
        ax.plot(val_steps, val_losses, color="#DC2626", marker="o", markersize=6, linewidth=2.2, label="Validation Loss (Unseen)", zorder=4)
        for s, l in zip(val_steps, val_losses):
            ax.annotate(f"{l:.4f}", (s, l), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color="#991B1B")

    ax.set_title("SmolVLA Multitask Training vs Validation Loss Curve", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Flow Matching Loss", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", loc="upper right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 학습 곡선 그래프 저장 완료: {save_path}")


if __name__ == "__main__":
    parse_and_plot()
