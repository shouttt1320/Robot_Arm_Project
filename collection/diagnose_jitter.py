"""SO-101 실시간 추론 떨림 및 제어 오차 정밀 진단기 (Jitter & Latency Diagnoser).

deploy_real_robot.py에서 수집된 동기화 로그(.npz)를 로드하여:
  1. 모델 순수 출력(action_raw) vs 시간축 앙상블(action_agg)의 변동성 비교
  2. 전송 명령(action_sent[t-1])과 실제 모터 반응(qpos[t]) 간의 시간 정렬 추종 오차
  3. 관절별 가동 범위 안전 클램핑(Clamp) 발생 빈도
  4. 제어 주기(30Hz) 준수율 및 연산/통신 지연 시간
을 정량 통계 테이블로 출력하고, 6개 관절별 정밀 궤적 그래프를 PNG로 생성합니다.
"""

import argparse
import glob
import os
import sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINT_NAMES = [
    "Joint 1 (Base)",
    "Joint 2 (Shoulder)",
    "Joint 3 (Elbow)",
    "Joint 4 (Wrist Pitch)",
    "Joint 5 (Wrist Roll)",
    "Joint 6 (Gripper)",
]


def analyze_and_plot_diagnostics(
    npz_path: str = "logs/latest_deploy_diag.npz",
    output_png: str = "logs/deploy_jitter_diagnosis.png",
    history_png: str = None,
):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"진단 로그 파일이 없습니다: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    
    timestamp = data["timestamp"]
    loop_dt_ms = data["loop_dt_ms"]
    inference_ms = data["inference_ms"]
    
    qpos_in_deg = data["qpos_in_deg"]       # (T, 6)
    qpos_in_raw = data["qpos_in_raw"]       # (T, 6)
    action_raw_deg = data["action_raw_deg"] # (T, 6)
    action_agg_deg = data["action_agg_deg"] # (T, 6)
    action_sent_deg = data["action_sent_deg"] # (T, 6)
    action_sent_raw = data["action_sent_raw"] # (T, 6)
    is_clamped = data["is_clamped"]         # (T, 6) bool
    
    T = len(timestamp)
    if T < 5:
        print(f"⚠️ 기록된 스텝 수가 너무 적습니다 (총 {T}스텝). 최소 5스텝 이상 주행 후 분석하세요.")
        return

    # 1. 시간 정렬(Time-Alignment): t-1 시점 명령과 t 시점 실제 모터 응답 매칭
    # 스텝 t(1~T-1)에서의 실제 위치 qpos_in_deg[t]는 스텝 t-1에서 보낸 action_sent_deg[t-1]의 결과임
    cmd_aligned = action_sent_deg[:-1]  # (T-1, 6)
    act_aligned = qpos_in_deg[1:]       # (T-1, 6)
    tracking_err = np.abs(cmd_aligned - act_aligned)  # (T-1, 6)
    
    # 2. 스텝 간 변동성(Jitter: Frame-to-frame delta)
    delta_raw = np.abs(np.diff(action_raw_deg, axis=0))  # (T-1, 6)
    delta_agg = np.abs(np.diff(action_agg_deg, axis=0))  # (T-1, 6)
    
    mean_tracking_err = tracking_err.mean(axis=0)
    max_tracking_err = tracking_err.max(axis=0)
    mean_delta_raw = delta_raw.mean(axis=0)
    mean_delta_agg = delta_agg.mean(axis=0)
    clamp_counts = is_clamped.sum(axis=0)

    # 3. 타이밍 통계
    avg_loop_dt = np.mean(loop_dt_ms)
    max_loop_dt = np.max(loop_dt_ms)
    avg_inf_ms = np.mean(inference_ms)
    over_33ms_count = np.sum(loop_dt_ms > 34.0)

    # 4. 터미널 리포트 출력
    print("\n" + "=" * 78)
    print(f" 🔍 [SO-101 실시간 추론 정밀 진단 리포트] (총 {T} 스텝, {timestamp[-1]-timestamp[0]:.1f}초 주행)")
    print(f" 📁 분석 로그: {npz_path}")
    print("=" * 78)
    print(f" • 제어 주기   : 평균 {avg_loop_dt:.1f} ms (목표: 33.3 ms @ 30Hz) | 최대 {max_loop_dt:.1f} ms")
    print(f" • 모델 추론   : 평균 {avg_inf_ms:.1f} ms (순수 ACT Forward 타임)")
    print(f" • 33ms 초과   : {over_33ms_count} / {T} 스텝 ({(over_33ms_count/T)*100:.1f}%)")
    print("-" * 78)
    print(f"{'관절 명칭':<20} | {'Raw 변동(Δ)':<11} | {'Agg 변동(Δ)':<11} | {'추종오차(평균/최대)':<18} | {'클램프(한계)'}")
    print("-" * 78)
    for j in range(6):
        print(
            f"{JOINT_NAMES[j]:<20} | {mean_delta_raw[j]:5.2f}°       | {mean_delta_agg[j]:5.2f}°       | "
            f"{mean_tracking_err[j]:5.2f}° / {max_tracking_err[j]:5.2f}°    | {clamp_counts[j]:3d}회"
        )
    print("-" * 78)

    # 가이드 안내
    guides = []
    if np.any(clamp_counts > 0):
        clamped_j = [JOINT_NAMES[j] for j in range(6) if clamp_counts[j] > 0]
        guides.append(f"⚠️ 클램프 발생 관절: {', '.join(clamped_j)} ➔ calibration.json의 min/max 한계값 확인 권장")
    if over_33ms_count > T * 0.1:
        guides.append("⚠️ 제어 주기가 33ms를 자주 초과합니다 ➔ 프리뷰 창 비활성화(--no-preview) 또는 통신 점검 권장")
    if np.any(mean_tracking_err > 5.0):
        lagged_j = [JOINT_NAMES[j] for j in range(6) if mean_tracking_err[j] > 5.0]
        guides.append(f"⚠️ 모터 추종 지연(오차 > 5°): {', '.join(lagged_j)} ➔ 모터 부하, 토크 또는 속도 프로파일 확인 권장")
    if not guides:
        guides.append("✅ 모든 관절의 추종 오차 및 제어 주기가 양호한 범위 내에 있습니다.")
    
    print("💡 판정 가이드:")
    for g in guides:
        print(f"  {g}")
    print("=" * 78 + "\n")

    # 5. 고해상도 다채널 궤적 그래프 생성 (3행 2열: 6개 관절)
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
    axes = axes.flatten()
    steps = np.arange(T)

    for j in range(6):
        ax = axes[j]
        # 1. 모델 순수 예측 (Raw 0번 액션)
        ax.plot(steps, action_raw_deg[:, j], label="Model Raw (Instant #0)", color="#999999", linestyle=":", linewidth=1.4, alpha=0.8)
        # 2. 전송 명령 (Temporal Aggregated & Clamped Target)
        ax.plot(steps, action_sent_deg[:, j], label="Command Sent (Target)", color="#1f77b4", linewidth=2.0)
        # 3. 실제 모터 반응 (Actual Encoder Reading)
        ax.plot(steps, qpos_in_deg[:, j], label="Motor Feedback (Actual)", color="#ff7f0e", linewidth=1.8, alpha=0.9)

        # 4. 클램프 발생 시점 빨간색 점 표기
        clamped_idx = np.where(is_clamped[:, j])[0]
        if len(clamped_idx) > 0:
            ax.scatter(clamped_idx, action_sent_deg[clamped_idx, j], color="red", s=35, zorder=5, label="Clamped Limit")

        ax.set_title(
            f"{JOINT_NAMES[j]} [Avg Error: {mean_tracking_err[j]:.1f}° | Jitter Raw: {mean_delta_raw[j]:.2f}° ➔ Agg: {mean_delta_agg[j]:.2f}°]",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Angle (Degree)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if j == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=9)

    axes[4].set_xlabel("Time Step (30Hz)", fontsize=11)
    axes[5].set_xlabel("Time Step (30Hz)", fontsize=11)

    fig.suptitle(
        f"SO-101 Real-Time Execution Jitter & Tracking Diagnosis [Steps: {T} | Loop: {avg_loop_dt:.1f}ms]",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=160)
    if history_png:
        os.makedirs(os.path.dirname(os.path.abspath(history_png)), exist_ok=True)
        plt.savefig(history_png, dpi=160)
    plt.close(fig)

    print(f"📊 진단 그래프 저장 완료 ➔ {output_png}")
    if history_png:
        print(f"💾 [진단 그래프 히스토리 보존] ➔ {history_png}")
    print()


def save_visual_trajectory_summary(
    keyframes: list,
    output_png: str = "logs/latest_deploy_visual.png",
    history_png: str = None,
    frames_dir: str = "logs/latest_frames",
):
    """실시간 추론 시 샘플링된 카메라 영상들을 시계열 접촉 시트(Contact Sheet) 및 개별 고화질 파일로 저장."""
    if not keyframes:
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # 이전 개별 프레임 정리
    for f in glob.glob(os.path.join(frames_dir, "*.jpg")):
        try:
            os.remove(f)
        except OSError:
            pass

    # 1. 개별 고화질 프레임 저장 (정밀 줌인 분석용)
    for kf in keyframes:
        t = kf["step"]
        top_img = kf.get("top")
        wrist_img = kf.get("wrist")
        if top_img is not None:
            cv2.imwrite(os.path.join(frames_dir, f"step_{t:03d}_top.jpg"), top_img)
        if wrist_img is not None:
            cv2.imwrite(os.path.join(frames_dir, f"step_{t:03d}_wrist.jpg"), wrist_img)

    # 2. 접촉 시트(Contact Sheet) 생성: 최대 8개 대표 스텝 균등 샘플링
    n_frames = len(keyframes)
    if n_frames > 8:
        indices = np.linspace(0, n_frames - 1, 8, dtype=int)
        sampled = [keyframes[i] for i in indices]
    else:
        sampled = keyframes

    cell_w, cell_h = 320, 240
    top_row = []
    wrist_row = []

    for kf in sampled:
        t = kf["step"]
        q_base = kf["qpos_deg"][0]
        q_shldr = kf["qpos_deg"][1]
        tgt_base = kf["target_deg"][0]
        tgt_shldr = kf["target_deg"][1]

        top_cell = cv2.resize(kf["top"], (cell_w, cell_h)) if kf.get("top") is not None else np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        wrist_cell = cv2.resize(kf["wrist"], (cell_w, cell_h)) if kf.get("wrist") is not None else np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

        # 상단 HUD (스텝 번호 및 경과 시간)
        cv2.rectangle(top_cell, (0, 0), (cell_w, 24), (20, 20, 20), -1)
        cv2.putText(top_cell, f"Step {t:3d} ({t*33.3/1000.0:3.1f}s)", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 하단 HUD (베이스 및 숄더 각도)
        cv2.rectangle(top_cell, (0, cell_h - 22), (cell_w, cell_h), (0, 0, 0), -1)
        cv2.putText(top_cell, f"Base:{q_base:+.1f} Shldr:{q_shldr:+.1f}", (4, cell_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1)

        # Wrist 뷰 HUD
        cv2.rectangle(wrist_cell, (0, 0), (cell_w, 24), (20, 20, 20), -1)
        cv2.putText(wrist_cell, f"Wrist Cam @ Step {t:3d}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        cv2.rectangle(wrist_cell, (0, cell_h - 22), (cell_w, cell_h), (0, 0, 0), -1)
        cv2.putText(wrist_cell, f"Tgt B:{tgt_base:+.1f} S:{tgt_shldr:+.1f}", (4, cell_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 1)

        # 경계선 테두리
        top_cell = cv2.copyMakeBorder(top_cell, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(80, 80, 80))
        wrist_cell = cv2.copyMakeBorder(wrist_cell, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(80, 80, 80))

        top_row.append(top_cell)
        wrist_row.append(wrist_cell)

    contact_top = np.hstack(top_row)
    contact_wrist = np.hstack(wrist_row)
    grid = np.vstack([contact_top, contact_wrist])

    # 타이틀 배너 추가
    banner_h = 36
    banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, "SO-101 REAL-TIME DEPLOYMENT VISUAL TIMELINE [Top: Top Cam | Bottom: Wrist Cam]", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    final_canvas = np.vstack([banner, grid])

    cv2.imwrite(output_png, final_canvas)
    if history_png:
        os.makedirs(os.path.dirname(os.path.abspath(history_png)), exist_ok=True)
        cv2.imwrite(history_png, final_canvas)
    print(f"📸 [시각적 추론 타임라인 저장 완료] ➔ {output_png}")
    if history_png:
        print(f"💾 [시각 타임라인 히스토리 보존] ➔ {history_png}")
    print(f"📁 [스텝별 개별 고화질 사진 저장 완료] ➔ {frames_dir}/ ({len(keyframes)*2}장)")


def main():
    parser = argparse.ArgumentParser(description="SO-101 실시간 추론 떨림 및 제어 오차 정밀 진단 스크립트")
    parser.add_argument("--log", type=str, default="logs/latest_deploy_diag.npz", help="분석할 .npz 로그 파일 경로")
    parser.add_argument("--output", type=str, default="logs/deploy_jitter_diagnosis.png", help="저장할 결과 그래프 파일명")
    args = parser.parse_args()

    analyze_and_plot_diagnostics(npz_path=args.log, output_png=args.output)


if __name__ == "__main__":
    main()
