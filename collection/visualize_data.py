"""SO-101 픽앤플레이스 HDF5 데이터셋 검증 및 시각화 도구 (Phase 4).

수집된 에피소드 데이터(Top/Wrist 영상, qpos, action, qvel)의 무결성을 검증하고,
동영상(MP4), 관절 궤적 그래프(PNG), 대화형 플레이어로 데이터 품질을 확인합니다.

사용법:
    # 1. 특정 에피소드(0번) 검증 + 영상/그래프 저장
    python3 visualize_data.py --episode 0

    # 2. 전체 데이터셋 무결성 일괄 검증 (리포트 테이블 출력)
    python3 visualize_data.py --all

    # 3. 실시간 대화형 재생 플레이어 실행 (스페이스바: 재생/일시정지, 좌우 화살표: 프레임 탐색)
    python3 visualize_data.py --episode 0 --play
"""

import argparse
import glob
import os
import sys
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

# 관절 명칭 정의
JOINT_NAMES = [
    "Joint 1 (Base)",
    "Joint 2 (Shoulder)",
    "Joint 3 (Elbow)",
    "Joint 4 (Wrist Pitch)",
    "Joint 5 (Wrist Roll)",
    "Joint 6 (Gripper)",
]


def load_episode(hdf5_path: str):
    """HDF5 에피소드 파일 로드"""
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {hdf5_path}")

    data = {}
    with h5py.File(hdf5_path, "r") as root:
        data["sim"] = root.attrs.get("sim", False)
        actual_ts = root.attrs.get("actual_timesteps", None)
        if actual_ts is None:
            actual_ts = root.attrs.get("total_timesteps", None)
        data["actual_timesteps"] = actual_ts
        data["max_timesteps"] = root.attrs.get("max_timesteps", None)

        data["action"] = root["action"][()]  # (T, 6)
        data["qpos"] = root["observations/qpos"][()]  # (T, 6)
        data["qvel"] = root["observations/qvel"][()]  # (T, 6)

        data["images"] = {}
        if "observations/images" in root:
            for cam_name in root["observations/images"].keys():
                data["images"][cam_name] = root[f"observations/images/{cam_name}"][()]

    return data


def verify_episode(hdf5_path: str, verbose: bool = True) -> dict:
    """단일 에피소드의 무결성 검증 (NaN, 차원, 추종 오차, 프레임 일관성 검사)"""
    file_size_mb = os.path.getsize(hdf5_path) / (1024 * 1024)
    filename = os.path.basename(hdf5_path)

    issues = []
    warnings = []

    try:
        data = load_episode(hdf5_path)
    except Exception as e:
        return {
            "filename": filename,
            "status": "FAIL",
            "message": f"파일 로드 오류: {e}",
            "size_mb": file_size_mb,
        }

    qpos = data["qpos"]
    action = data["action"]
    qvel = data["qvel"]
    images = data["images"]

    num_steps = len(qpos)

    # 1. 텐서 형태 및 크기 일관성 검사
    if action.shape[0] != num_steps or qvel.shape[0] != num_steps:
        issues.append(f"스텝 수 불일치: qpos({num_steps}) vs action({action.shape[0]})")

    if qpos.shape[1] != 6 or action.shape[1] != 6:
        issues.append(f"관절 차원 오류: qpos {qpos.shape[1]}차원, action {action.shape[1]}차원 (기대값: 6)")

    # 2. NaN / Inf 검사
    if np.isnan(qpos).any() or np.isnan(action).any() or np.isnan(qvel).any():
        issues.append("qpos 또는 action에 NaN(결측치) 존재")
    if np.isinf(qpos).any() or np.isinf(action).any():
        issues.append("qpos 또는 action에 Inf(무한대) 존재")

    # 3. 카메라 프레임 검사
    for cam_name, img_seq in images.items():
        if img_seq.shape[0] != num_steps:
            issues.append(f"카메라 '{cam_name}' 프레임 수 불일치 ({img_seq.shape[0]} != {num_steps})")
        if img_seq.shape[1:] != (480, 640, 3):
            warnings.append(f"카메라 '{cam_name}' 해상도 비표준: {img_seq.shape[1:]}")
        # 완전 블랙 프레임 검사
        if img_seq[0].mean() < 1.0:
            warnings.append(f"카메라 '{cam_name}' 시작 화면이 완전히 검은색입니다.")

    # 4. 리더-팔로워 추종 오차 분석
    tracking_errors = np.abs(qpos - action)
    mean_err_deg = np.rad2deg(np.mean(tracking_errors))
    max_err_deg = np.rad2deg(np.max(tracking_errors))

    # 추종 오차가 25도 이상 벌어지면 경고 (모터 걸림 or 부하 의심)
    if max_err_deg > 30.0:
        warnings.append(f"최대 관절 추종 오차 높음: {max_err_deg:.1f}° (모터 부하 확인 필요)")

    # 5. 관절 동작 범위 (작업이 정지된 에피소드인지 검사)
    qpos_range = np.ptp(qpos, axis=0)  # max - min
    if np.all(qpos_range < 0.05):
        warnings.append("로봇이 거의 움직이지 않은 정적(Static) 에피소드입니다.")

    status = "FAIL" if issues else ("WARN" if warnings else "PASS")

    result = {
        "filename": filename,
        "status": status,
        "num_steps": num_steps,
        "actual_timesteps": data.get("actual_timesteps") or num_steps,
        "size_mb": file_size_mb,
        "mean_err_deg": mean_err_deg,
        "max_err_deg": max_err_deg,
        "cams": list(images.keys()),
        "issues": issues,
        "warnings": warnings,
    }

    if verbose:
        status_icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
        print("\n" + "=" * 65)
        print(f" {status_icons[status]} [에피소드 검증 결과] {filename}")
        print("=" * 65)
        print(f" - 데이터 상태: {status}")
        print(f" - 전체 스텝 수: {num_steps} (실제 조작: {result['actual_timesteps']} 스텝)")
        print(f" - 파일 용량: {file_size_mb:.2f} MB")
        print(f" - 수집 카메라: {result['cams']}")
        print(f" - 평균 추종 오차: {mean_err_deg:.2f}° / 최대 추종 오차: {max_err_deg:.2f}°")

        if issues:
            print("\n 🚨 [치명적 결함 (학습 불가)]:")
            for iss in issues:
                print(f"   • {iss}")
        if warnings:
            print("\n ⚠️ [경고 사항 (품질 확인 권장)]:")
            for w in warnings:
                print(f"   • {w}")
        if not issues and not warnings:
            print("\n 🎉 모든 무결성 검사를 완벽히 통과했습니다! (ACT 학습에 즉시 사용 가능)")
        print("=" * 65)

    return result


def export_verification_video(data: dict, output_path: str, fps: int = 30):
    """Top + Wrist 영상을 가로로 합성하고 HUD 오버레이를 포함하여 MP4 비디오로 저장"""
    images = data["images"]
    if not images:
        print("[경고] 저장할 카메라 이미지가 없습니다.")
        return

    cam_names = sorted(list(images.keys()))
    num_frames = data["qpos"].shape[0]

    # 프레임 크기 결정
    h, w = images[cam_names[0]][0].shape[:2]
    total_w = w * len(cam_names)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 표준 H.264(avc1) 코덱 우선 시도 (일반 플레이어 및 브라우저 완벽 호환)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, h))

    qpos = data["qpos"]
    action = data["action"]

    for t in range(num_frames):
        cam_tiles = []
        for name in cam_names:
            # HDF5는 RGB이므로 OpenCV 저장을 위해 BGR로 변환
            rgb = images[name][t]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # 카메라 이름 라벨
            cv2.putText(
                bgr, f"CAM: {name.upper()}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            cam_tiles.append(bgr)

        combined = np.hstack(cam_tiles)

        # 하단 HUD 오버레이 (시간, 스텝, 오차 정보)
        time_sec = t / float(fps)
        cv2.rectangle(combined, (0, h - 45), (total_w, h), (30, 30, 30), -1)

        err_deg = np.rad2deg(np.mean(np.abs(qpos[t] - action[t])))
        grip_state = f"Grip: {qpos[t, 5]:.2f} rad"

        hud_text = f"Step: {t+1:3d}/{num_frames} ({time_sec:4.1f}s) | Lag: {err_deg:4.1f}deg | {grip_state}"
        cv2.putText(
            combined, hud_text, (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        out.write(combined)

    out.release()
    print(f"🎥 검증 동영상 저장 완료: {output_path}")


def export_trajectory_plot(data: dict, output_path: str, fps: int = 30):
    """6개 관절의 qpos(팔로워) vs action(리더 목표) 시계열 궤적 비교 그래프 저장"""
    qpos = data["qpos"]
    action = data["action"]
    qvel = data["qvel"]
    num_steps = len(qpos)
    time_axis = np.arange(num_steps) / float(fps)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("SO-101 Joint Trajectory Verification (qpos vs action)", fontsize=14, fontweight="bold")

    for i in range(6):
        ax = axs[i]
        # 라디안을 도(deg) 단위로 변환하여 시각화
        qpos_deg = np.rad2deg(qpos[:, i])
        action_deg = np.rad2deg(action[:, i])

        ax.plot(time_axis, qpos_deg, label="Follower State (qpos)", color="#1f77b4", linewidth=2.0)
        ax.plot(time_axis, action_deg, label="Leader Command (action)", color="#ff7f0e", linestyle="--", linewidth=1.5)

        # 오차 영역 음영 표시
        ax.fill_between(time_axis, qpos_deg, action_deg, color="#d62728", alpha=0.15, label="Tracking Error")

        ax.set_ylabel(f"{JOINT_NAMES[i]}\n(deg)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    axs[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"📊 관절 궤적 그래프 저장 완료: {output_path}")


def interactive_player(data: dict, fps: int = 30):
    """OpenCV 대화형 비디오 플레이어 (스페이스바: 재생/정지, 화살표: 프레임 탐색)"""
    images = data["images"]
    if not images:
        print("[경고] 표시할 카메라 이미지가 없습니다.")
        return

    cam_names = sorted(list(images.keys()))
    num_frames = data["qpos"].shape[0]
    curr_frame = 0
    playing = True
    delay = int(1000 / fps)

    window_name = "SO-101 Episode Interactive Inspector [Space: Play/Pause, Arrows: Step, Q: Quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 65)
    print(" 🎮 [대화형 검증 플레이어 조작법]")
    print("  • [Space] : 재생 / 일시정지 토글")
    print("  • [D / Right Arrow] : +1 프레임 이동")
    print("  • [A / Left Arrow]  : -1 프레임 이동")
    print("  • [R] : 처음으로 리셋")
    print("  • [Q / ESC] : 플레이어 종료")
    print("=" * 65 + "\n")

    while True:
        cam_tiles = []
        for name in cam_names:
            rgb = images[name][curr_frame]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr, f"CAM: {name.upper()}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            cam_tiles.append(bgr)

        combined = np.hstack(cam_tiles)
        h, total_w = combined.shape[:2]

        # 상태 오버레이
        time_sec = curr_frame / float(fps)
        status_str = "PLAYING" if playing else "PAUSED"
        cv2.rectangle(combined, (0, h - 45), (total_w, h), (40, 40, 40), -1)
        hud_text = f"[{status_str}] Frame {curr_frame+1:3d}/{num_frames} ({time_sec:4.1f}s) | Space: Toggle | Q: Exit"
        cv2.putText(
            combined, hud_text, (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(delay if playing else 0) & 0xFF
        if key in [ord("q"), 27]:  # Q or ESC
            break
        elif key == ord(" "):  # Space
            playing = not playing
        elif key in [ord("d"), 83]:  # Right / D
            curr_frame = min(num_frames - 1, curr_frame + 1)
        elif key in [ord("a"), 81]:  # Left / A
            curr_frame = max(0, curr_frame - 1)
        elif key == ord("r"):  # Reset
            curr_frame = 0

        if playing:
            curr_frame += 1
            if curr_frame >= num_frames:
                curr_frame = 0  # 루프 재생

    cv2.destroyAllWindows()


def run_all_verification(data_dir: str, task: str):
    """전체 데이터셋 일괄 검증 및 종합 리포트 출력"""
    task_dir = os.path.join(data_dir, task)
    episodes = sorted(glob.glob(os.path.join(task_dir, "episode_*.hdf5")))

    if not episodes:
        print(f"❌ '{task_dir}' 경로에 episode_*.hdf5 파일이 없습니다.")
        return

    print("\n" + "=" * 90)
    print(f" 📋 [데이터셋 일괄 검증 리포트] 태스크: {task} (총 {len(episodes)}개 에피소드)")
    print("=" * 90)
    print(f"{'파일명':<18} | {'상태':<6} | {'총스텝':<7} | {'실스텝':<7} | {'용량(MB)':<9} | {'최대오차':<8} | {'카메라':<12}")
    print("-" * 90)

    pass_count = 0
    warn_count = 0
    fail_count = 0
    total_steps = 0

    for ep_path in episodes:
        res = verify_episode(ep_path, verbose=False)
        st = res["status"]
        if st == "PASS":
            pass_count += 1
            icon = "✅ PASS"
        elif st == "WARN":
            warn_count += 1
            icon = "⚠️ WARN"
        else:
            fail_count += 1
            icon = "❌ FAIL"

        steps = res.get("num_steps", 0)
        actual = res.get("actual_timesteps") or steps
        size = res.get("size_mb", 0.0)
        max_err = f"{res.get('max_err_deg', 0.0):.1f}°"
        cams = ",".join(res.get("cams", []))
        total_steps += steps

        print(f"{res['filename']:<18} | {icon:<6} | {steps:<7} | {str(actual):<7} | {size:<9.1f} | {max_err:<8} | {cams:<12}")

    print("=" * 90)
    print(f" 📊 검증 요약: 총 {len(episodes)}개 중 정상(PASS) {pass_count}개, 경고(WARN) {warn_count}개, 결함(FAIL) {fail_count}개")
    print(f"    총 누적 학습 프레임: {total_steps} 스텝 (약 {total_steps / 30.0 / 60.0:.1f} 분 분량)")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SO-101 픽앤플레이스 HDF5 데이터 검증 및 시각화 도구")
    parser.add_argument("--task", type=str, default="so101_pick_place", help="태스크 이름 (기본값: so101_pick_place)")
    parser.add_argument("--data-dir", type=str, default="data", help="데이터 루트 경로 (기본값: data)")
    parser.add_argument("--episode", type=int, default=0, help="검증할 에피소드 인덱스 (기본값: 0)")
    parser.add_argument("--all", action="store_true", help="전체 에피소드 일괄 무결성 검증 리포트 출력")
    parser.add_argument("--play", action="store_true", help="대화형 GUI 비디오 플레이어 실행")
    parser.add_argument("--no-video", action="store_true", help="MP4 동영상 저장 비활성화")
    parser.add_argument("--no-plot", action="store_true", help="관절 궤적 그래프 PNG 저장 비활성화")
    parser.add_argument("--check-only", action="store_true", help="파일 저장 없이 무결성만 신속 검증")
    args = parser.parse_args()

    if args.all:
        run_all_verification(args.data_dir, args.task)
        return

    task_dir = os.path.join(args.data_dir, args.task)
    ep_file = f"episode_{args.episode}.hdf5"
    ep_path = os.path.join(task_dir, ep_file)

    if not os.path.exists(ep_path):
        print(f"❌ 에피소드 파일을 찾을 수 없습니다: {ep_path}")
        print(f"   먼저 'python3 record_episodes.py' 로 데이터를 녹화하세요.")
        sys.exit(1)

    # 1. 무결성 검증
    res = verify_episode(ep_path, verbose=True)
    if args.check_only:
        return

    # 데이터 로드
    data = load_episode(ep_path)
    vis_dir = os.path.join(task_dir, "visualizations")

    # 2. 검증 동영상 저장
    if not args.no_video:
        video_path = os.path.join(vis_dir, f"episode_{args.episode}_preview.mp4")
        export_verification_video(data, video_path)

    # 3. 관절 궤적 그래프 저장
    if not args.no_plot:
        plot_path = os.path.join(vis_dir, f"episode_{args.episode}_traj.png")
        export_trajectory_plot(data, plot_path)

    # 4. 대화형 플레이어 실행 (옵션)
    if args.play:
        interactive_player(data)


if __name__ == "__main__":
    main()
