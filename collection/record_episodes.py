"""SO-101 픽앤플레이스(Pick-and-Place) 모방학습 데이터셋 수집 도구 (Phase 4).

시간 제한 없이 [스페이스바]로 자유롭게 시작/종료할 수 있는 실시간 텔레옵 및 데이터 녹화 도구입니다.
카메라 프리뷰 창 또는 터미널 어디서든 [스페이스바]를 누르면 즉시 녹화가 시작/종료됩니다.
"""

import argparse
import atexit
import glob
import json
import os
import select
import signal
import sys
import termios
import time
import tty
import cv2
import h5py
import numpy as np

# 프로젝트 루트를 sys.path에 추가하여 어디서 실행해도 안전하게 임포트 지원
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from hardware.camera_interface import MultiCameraCapture
    from hardware.sts3215_driver import STS3215Driver
except ImportError:
    from camera_interface import MultiCameraCapture
    from sts3215_driver import STS3215Driver


class RawTerminalInput:
    """터미널에서 엔터 없이 단일 키(스페이스바 등)를 즉각 감지하기 위한 논블로킹 키보드 리더"""

    def __init__(self):
        self.is_tty = sys.stdin.isatty()
        self.old_settings = None
        self.enable()
        atexit.register(self.restore)

    def enable(self):
        if self.is_tty and self.old_settings is None:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.is_tty = False

    def restore(self):
        if self.is_tty and self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass
            self.old_settings = None

    def get_key(self) -> str:
        if not self.is_tty:
            return ""
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
            if rlist:
                return sys.stdin.read(1)
        except Exception:
            pass
        return ""


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def raw_to_rad(raw_val: int) -> float:
    """STS3215 Raw 값 (0~4095, 360도)을 라디안(-pi ~ +pi)으로 변환"""
    return (raw_val - 2048) * (2.0 * np.pi / 4096.0)


class EpisodeRecorder:

    def __init__(
        self,
        task_name: str = "so101_pick_place",
        data_dir: str = "data",
        config_path: str = "configs/calibration.json",
        top_cam_id: int = 4,
        wrist_cam_id: int = 6,
        fps: int = 30,
        show_preview: bool = True,
    ):
        self.task_name = task_name
        self.output_dir = os.path.join(data_dir, task_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.fps = fps
        self.dt = 1.0 / fps
        self.running = False
        self._stopped = False
        self.show_preview = show_preview and bool(os.environ.get("DISPLAY"))
        self.kb = RawTerminalInput()

        # 1. 캘리브레이션 로드
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 가 없습니다. 먼저 calibrate.py를 실행하세요.")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.leader_port = self.config["leader_port"]
        self.follower_port = self.config["follower_port"]
        self.joints_cfg = self.config["joints"]
        self.gripper_cfg = self.config["gripper"]

        # 2. 로봇 드라이버 초기화
        self.l_driver = STS3215Driver(
            self.leader_port, baudrate=1000000, timeout=0.03
        )
        self.f_driver = STS3215Driver(
            self.follower_port, baudrate=1000000, timeout=0.03
        )

        # 3. 카메라 설정
        self.cam_dict = {}
        if top_cam_id >= 0:
            self.cam_dict["top"] = top_cam_id
        if wrist_cam_id >= 0:
            self.cam_dict["wrist"] = wrist_cam_id

        if not self.cam_dict:
            raise ValueError("적어도 1개 이상의 카메라가 지정되어야 합니다!")

        self.camera_manager = MultiCameraCapture(
            self.cam_dict, width=640, height=480, fps=fps
        )

    def connect(self):
        print("\n" + "=" * 68)
        print(f" 🤖 [SO-101 데이터 수집기 (무제한 시간 & 스페이스바 제어)]")
        print(f" 📁 저장 디렉토리: {os.path.abspath(self.output_dir)}")
        print(f" 🎥 활성 카메라: {list(self.cam_dict.keys())} (FPS: {self.fps})")
        print(f" ⌨️ 조작법: [스페이스바] 한 번 누르면 시작, 완료 후 다시 누르면 종료!")
        print("=" * 68)

        # 로봇 연결
        if not self.l_driver.connect():
            raise ConnectionError(f"Leader 로봇 ({self.leader_port}) 연결 실패")
        if not self.f_driver.connect():
            self.l_driver.close()
            raise ConnectionError(
                f"Follower 로봇 ({self.follower_port}) 연결 실패"
            )

        self.l_driver.set_all_torques(range(1, 7), False)  # 리더 수동
        self.f_driver.set_all_torques(range(1, 7), True)  # 팔로워 토크 켬
        print("✅ 로봇 하드웨어 연결 완료 (리더: 수동 조작, 팔로워: 텔레옵 대기)")

        # 카메라 연결
        self.camera_manager.start()
        print("✅ 모든 카메라 실시간 캡처 준비 완료")

    def map_leader_to_follower(self, leader_pos: dict) -> dict:
        """리더 위치를 팔로워 목표 위치로 변환"""
        follower_goals = {}
        for i in range(1, 6):
            cfg = self.joints_cfg[f"joint_{i}"]
            l_val = leader_pos.get(i)
            if l_val is None:
                continue
            delta = (l_val - cfg["leader_zero"]) * cfg.get("direction", 1)
            target = cfg["follower_zero"] + delta
            min_lim = cfg.get("follower_min", cfg.get("min", 0))
            max_lim = cfg.get("follower_max", cfg.get("max", 4095))
            follower_goals[i] = int(clamp(target, min_lim, max_lim))

        l_grip = leader_pos.get(6)
        if l_grip is not None:
            l_open = self.gripper_cfg["leader_open"]
            l_close = self.gripper_cfg["leader_close"]
            f_open = self.gripper_cfg["follower_open"]
            f_close = self.gripper_cfg["follower_close"]

            denom = l_open - l_close
            ratio = (
                (l_grip - l_close) / denom if denom != 0 else 0.5
            )
            ratio = clamp(ratio, 0.0, 1.0)
            f_grip_target = f_close + ratio * (f_open - f_close)
            grip_min = min(f_open, f_close)
            grip_max = max(f_open, f_close)
            follower_goals[6] = int(clamp(f_grip_target, grip_min, grip_max))

        return follower_goals

    def get_next_episode_idx(self) -> int:
        """기존 저장된 episode_*.hdf5 파일들을 검색하여 다음 번호 반환"""
        existing = glob.glob(os.path.join(self.output_dir, "episode_*.hdf5"))
        if not existing:
            return 0
        indices = []
        for path in existing:
            base = os.path.basename(path).replace("episode_", "").replace(".hdf5", "")
            try:
                indices.append(int(base))
            except ValueError:
                pass
        return max(indices) + 1 if indices else 0

    def _render_preview(self, frames: dict, status: str, step: int = 0, elapsed: float = 0.0) -> int:
        """카메라 프레임들을 가로로 결합하고 HUD 오버레이 렌더링. 누른 키 코드 반환."""
        if not self.show_preview:
            return 255

        cam_imgs = []
        for name in ["top", "wrist"]:
            if name in frames and frames[name] is not None:
                img = frames[name].copy()
                label = f"CAM: {name.upper()}"
                cv2.putText(img, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cam_imgs.append(img)

        if not cam_imgs:
            return 255

        combined = np.hstack(cam_imgs)
        h, w, _ = combined.shape

        if status == "RECORDING":
            # 빨간색 녹화 바
            bar_color = (0, 0, 210)
            time_str = time.strftime("%M:%S", time.gmtime(elapsed)) + f".{int((elapsed % 1) * 10)}"
            status_text = f"● REC [{step:4d} frames | {time_str}] - Press SPACEBAR to STOP"
        else:
            # 대기 바
            bar_color = (50, 50, 50)
            status_text = "STANDBY: Move robot to start pose, press SPACEBAR to RECORD"

        cv2.rectangle(combined, (0, h - 50), (w, h), bar_color, -1)
        cv2.putText(combined, status_text, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

        try:
            cv2.imshow("SO-101 Data Collection [Space: Start/Stop, Q: Quit]", combined)
            return cv2.waitKey(1) & 0xFF
        except Exception:
            self.show_preview = False
            return 255

    def check_user_input(self, gui_key: int) -> str:
        """GUI 창과 터미널 양쪽의 키 입력을 취합하여 이벤트 반환 ('SPACE', 'QUIT', '')"""
        term_ch = self.kb.get_key()

        # 스페이스바 또는 엔터
        if gui_key in [32, 13] or term_ch in [" ", "\n", "\r"]:
            return "SPACE"
        # 종료 키 (Q or ESC)
        if gui_key in [ord("q"), 27] or term_ch in ["q", "Q"]:
            return "QUIT"

        return ""

    def standby_loop(self, episode_idx: int) -> bool:
        """녹화 시작 전 실시간 텔레옵 대기 루프. 스페이스바 입력 시 True 반환."""
        self.kb.enable()
        print("\n" + "-" * 68)
        print(f"🎬 [Episode {episode_idx}] 녹화 대기 중 (실시간 텔레옵 동작 중)")
        print("  • 물체를 시작 위치에 두고, 리더 팔을 움직여 로봇을 초기 자세로 잡으세요.")
        print("  👉 준비되면 [스페이스바]를 누르세요. (녹화가 즉시 시작됩니다!)")
        print("     (전체 종료: 'q')")
        print("-" * 68)

        while self.running:
            # 1. 텔레옵 유지
            l_pos = self.l_driver.read_positions(range(1, 7))
            f_goals = self.map_leader_to_follower(l_pos)
            self.f_driver.sync_write_positions(f_goals, speed=0, acc=0)

            # 2. 카메라 프리뷰 화면 갱신
            cam_frames = self.camera_manager.read_all()
            gui_key = self._render_preview(cam_frames, status="STANDBY")

            # 3. 키 입력 확인
            event = self.check_user_input(gui_key)
            if event == "SPACE":
                return True
            elif event == "QUIT":
                self.running = False
                return False

            time.sleep(0.01)

        return False

    def record_single_episode(self, episode_idx: int) -> bool:
        """단일 에피소드 녹화 루프 (시간 제한 없음, 스페이스바로 종료)"""
        ready = self.standby_loop(episode_idx)
        if not ready or not self.running:
            return False

        recorded_images = {name: [] for name in self.cam_dict.keys()}
        recorded_qpos = []
        recorded_action = []

        print("\n🔴 [RECORDING 시작!] 시간 제한 없이 원하는 만큼 편안하게 조작하세요.")
        print("   👉 작업이 끝나면 [스페이스바]를 다시 누르세요!\n")

        loop_start_time = time.time()
        step = 0

        # 안전 상한선 (실수로 방치 시 30분 초과 방지)
        safety_max_steps = 30 * 60 * self.fps

        while self.running and step < safety_max_steps:
            t0 = time.time()
            step += 1
            elapsed = t0 - loop_start_time

            # 1. 카메라 프레임 읽기
            cam_frames = self.camera_manager.read_all()
            gui_key = self._render_preview(cam_frames, status="RECORDING", step=step, elapsed=elapsed)

            for name, frame in cam_frames.items():
                if frame is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    recorded_images[name].append(rgb_frame)
                else:
                    if recorded_images[name]:
                        recorded_images[name].append(recorded_images[name][-1])

            # 2. 리더 위치 읽기 및 팔로워 동기화
            l_pos = self.l_driver.read_positions(range(1, 7))
            f_goals = self.map_leader_to_follower(l_pos)
            self.f_driver.sync_write_positions(f_goals, speed=0, acc=0)

            # 3. 팔로워 현재 관절값 읽기 (qpos)
            f_pos = self.f_driver.read_positions(range(1, 7))
            curr_qpos = [
                raw_to_rad(f_pos.get(i, f_goals.get(i, 2048)))
                for i in range(1, 7)
            ]
            curr_action = [
                raw_to_rad(f_goals.get(i, 2048)) for i in range(1, 7)
            ]

            recorded_qpos.append(curr_qpos)
            recorded_action.append(curr_action)

            # 터미널 진행률 표시 (시:분:초 형식)
            time_str = time.strftime("%M:%S", time.gmtime(elapsed)) + f".{int((elapsed % 1) * 10)}"
            sys.stdout.write(
                f"\r  🔴 [녹화 중...] {step:4d} 프레임 ({time_str}) | 정지하려면 [스페이스바]"
            )
            sys.stdout.flush()

            # 4. 종료 스페이스바 입력 감지
            event = self.check_user_input(gui_key)
            if event in ["SPACE", "QUIT"]:
                if event == "QUIT":
                    self.running = False
                print(f"\n\n⏹️ [녹화 완료!] 총 {step} 프레임 ({elapsed:4.1f}초)")
                break

            # 제어 주기 유지
            step_time = time.time() - t0
            sleep_time = self.dt - step_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        total_steps = len(recorded_qpos)
        if total_steps < 10:
            print("⚠️ 녹화된 스텝이 너무 짧아(10스텝 미만) 저장하지 않습니다.")
            return False

        # 판정 시 터미널 입력을 일시 복원
        self.kb.restore()

        print("\n" + "=" * 55)
        user_input = (
            input(
                f"👉 [Episode {episode_idx}] 저장할까요? [Space/Y(저장) / n(폐기) / q(종료)]: "
            )
            .strip()
            .lower()
        )

        if user_input in ["q", "quit"]:
            self.running = False
            return False

        if user_input in ["", "y", "yes", " "]:
            save_path = os.path.join(self.output_dir, f"episode_{episode_idx}.hdf5")
            print(f"\n[저장 중...] ➔ {save_path}")

            with h5py.File(save_path, "w", rdcc_nbytes=1024**2 * 2) as root:
                root.attrs["sim"] = False
                root.attrs["total_timesteps"] = total_steps
                root.attrs["fps"] = self.fps

                # 1. Action 데이터셋 (T, 6)
                root.create_dataset(
                    "action",
                    data=np.array(recorded_action, dtype=np.float32),
                )

                # 2. Observations 그룹
                obs = root.create_group("observations")
                obs.create_dataset(
                    "qpos", data=np.array(recorded_qpos, dtype=np.float32)
                )

                # qvel 계산
                qpos_arr = np.array(recorded_qpos, dtype=np.float32)
                qvel_arr = np.zeros_like(qpos_arr)
                if len(qpos_arr) > 1:
                    qvel_arr[1:] = (qpos_arr[1:] - qpos_arr[:-1]) * self.fps
                obs.create_dataset("qvel", data=qvel_arr)

                # 이미지 그룹
                img_grp = obs.create_group("images")
                for cam_name in self.cam_dict.keys():
                    img_data = np.array(recorded_images[cam_name], dtype=np.uint8)
                    img_grp.create_dataset(
                        cam_name,
                        data=img_data,
                        chunks=(1, 480, 640, 3),
                        compression="gzip",
                        compression_opts=2,
                    )

            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"🎉 Episode {episode_idx} 저장 완료! (용량: {file_size_mb:.1f} MB, {total_steps} 프레임)")
            return True
        else:
            print(f"❌ Episode {episode_idx} 폐기됨. (저장하지 않고 다음 에피소드로 이동)")
            return False

    def run(self):
        self.connect()
        self.running = True

        try:
            while self.running:
                next_idx = self.get_next_episode_idx()
                print(f"\n📊 [데이터셋 진행률] 현재 저장된 에피소드 수: {next_idx} 개")
                saved = self.record_single_episode(next_idx)
        finally:
            self.stop()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        self.kb.restore()

        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        print("\n\n[안전 정지] Follower 및 Leader의 모든 모터 토크를 안전하게 해제합니다...")
        try:
            if self.f_driver and self.f_driver.ser and self.f_driver.ser.is_open:
                self.f_driver.set_all_torques(range(1, 7), False)
            if self.l_driver and self.l_driver.ser and self.l_driver.ser.is_open:
                self.l_driver.set_all_torques(range(1, 7), False)
            time.sleep(0.05)
        except Exception as err:
            print(f"[경고] 토크 해제 중 예외 발생: {err}")
        finally:
            if self.l_driver:
                self.l_driver.close()
            if self.f_driver:
                self.f_driver.close()

        try:
            self.camera_manager.stop()
        except Exception:
            pass

        print("✅ 모든 모터 토크 해제 및 카메라 종료 완료.")


def main():
    parser = argparse.ArgumentParser(
        description="SO-101 픽앤플레이스 데이터 수집기 (스페이스바 제어 & 무제한 시간)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="so101_pick_place",
        help="태스크 이름 (기본값: so101_pick_place)",
    )
    parser.add_argument(
        "--top-cam",
        type=int,
        default=4,
        help="외부 상단 카메라 인덱스 (기본값: 4, 사용안함: -1)",
    )
    parser.add_argument(
        "--wrist-cam",
        type=int,
        default=6,
        help="손목 카메라 인덱스 (기본값: 6, 사용안함: -1)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="데이터 수집 주기 FPS (기본값: 30Hz)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="실시간 카메라 프리뷰 창 비활성화 (터미널 전용 모드)",
    )
    args = parser.parse_args()

    recorder = EpisodeRecorder(
        task_name=args.task,
        top_cam_id=args.top_cam,
        wrist_cam_id=args.wrist_cam,
        fps=args.fps,
        show_preview=not args.no_preview,
    )

    def sigint_handler(sig, frame):
        recorder.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    recorder.run()


if __name__ == "__main__":
    main()
