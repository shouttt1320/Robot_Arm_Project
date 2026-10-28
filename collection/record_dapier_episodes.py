"""SO-101 DAPIER-ACT 전용 멀티모달 데이터 수집기.

수집 항목 (실시간 동기화):
1. Head Depth: OS30A (640x460 uint16 depth 맵 및 DP3 512점 포인트클라우드)
2. Left Wrist: 좌측 손목 카메라 RGB (240x320)
3. Right Wrist: 우측 손목 카메라 RGB (240x320)
4. Dual SO-101: 12개 관절 모터 각도 (Follower qpos, Leader target action)

조작법:
- [스페이스바] 한 번 누르면 에피소드 녹화 시작
- 작업 완료 후 [스페이스바] 다시 누르면 에피소드 저장 및 종료
"""

import argparse
import atexit
import glob
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import h5py
import numpy as np

# 프로젝트 루트 임포트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.camera_interface import MultiCameraCapture
from hardware.os30a_camera import OS30ACamera
from hardware.dual_arm_controller import DualArmController
from models.so101_dapier_act.config import DapierACTConfig


class RawTerminalInput:
    """터미널에서 엔터 없이 스페이스바를 감지하기 위한 키보드 리더"""

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


class DapierEpisodeRecorder:
    def __init__(
        self,
        task_name: str = "dapier_dual_arm_pick",
        data_dir: str = "data",
        fps: int = 20,
        left_wrist_cam_id: int = 4,
        right_wrist_cam_id: int = 6,
        left_follower_port: str = "/dev/ttyACM0",
        right_follower_port: str = "/dev/ttyACM1",
        left_leader_port: str = "/dev/ttyACM2",
        right_leader_port: str = "/dev/ttyACM3",
        use_sample_depth: bool = True,
        show_preview: bool = True,
    ):
        self.task_name = task_name
        self.output_dir = Path(data_dir) / task_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.dt = 1.0 / fps
        self.kb = RawTerminalInput()
        self.show_preview = show_preview and bool(os.environ.get("DISPLAY"))
        self.cfg = DapierACTConfig()

        print("=" * 70)
        print(" 🤖 [SO-101 DAPIER-ACT 멀티모달 데이터 수집기]")
        print(f" • 저장 디렉토리 : {self.output_dir.resolve()}")
        print(f" • 제어 주기     : {self.fps} Hz (dt={self.dt*1000:.1f}ms)")
        print("=" * 70)

        # 1. OS30A 뎁스 카메라 초기화
        print("\n📷 [1/3] OS30A 뎁스 카메라 초기화 중...")
        self.os30a = OS30ACamera(config=self.cfg, use_sample_fallback=use_sample_depth)

        # 2. 손목 RGB 카메라 초기화 (좌/우)
        print("📷 [2/3] 손목 RGB 카메라 초기화 중...")
        cam_dict = {"left_wrist": left_wrist_cam_id, "right_wrist": right_wrist_cam_id}
        self.wrist_cams = MultiCameraCapture(
            cam_dict, width=self.cfg.wrist_width, height=self.cfg.wrist_height, fps=fps
        )

        # 3. 양팔 로봇 제어기 초기화
        print("🦾 [3/3] 양팔 SO-101 (12 DOF) 로봇 암 연결 중...")
        self.arms = DualArmController(
            left_follower_port=left_follower_port,
            right_follower_port=right_follower_port,
            left_leader_port=left_leader_port,
            right_leader_port=right_leader_port,
        )
        self.arms.connect()

    def get_next_episode_index(self) -> int:
        files = glob.glob(str(self.output_dir / "episode_*.hdf5"))
        indices = []
        for f in files:
            stem = Path(f).stem
            try:
                idx = int(stem.split("_")[-1])
                indices.append(idx)
            except ValueError:
                pass
        return max(indices, default=-1) + 1

    def run(self):
        print("\n💡 [조작 안내]")
        print(" • [스페이스바] : 녹화 시작 / 저장 완료")
        print(" • [Q / ESC]    : 프로그램 종료")

        recording = False
        buffer_qpos = []
        buffer_action = []
        buffer_left = []
        buffer_right = []
        buffer_depth = []
        buffer_pointcloud = []

        try:
            while True:
                loop_start = time.time()

                # 키보드 입력 체크
                key = self.kb.get_key()
                if key in ["q", "Q", "\x1b"]:
                    print("\n🛑 사용자에 의해 프로그램이 종료되었습니다.")
                    break

                if key == " ":
                    if not recording:
                        recording = True
                        buffer_qpos.clear()
                        buffer_action.clear()
                        buffer_left.clear()
                        buffer_right.clear()
                        buffer_depth.clear()
                        buffer_pointcloud.clear()
                        print("\n🔴 >>> [녹화 시작!] 에피소드를 진행하세요...")
                    else:
                        recording = False
                        print(f"\n⏹️ <<< [녹화 종료] 총 {len(buffer_qpos)} 프레임 수집됨. 저장 중...")
                        self.save_episode(
                            buffer_qpos,
                            buffer_action,
                            buffer_left,
                            buffer_right,
                            buffer_depth,
                            buffer_pointcloud,
                        )

                # 1. 로봇 텔레옵 스텝
                follower_qpos, leader_action = self.arms.teleop_step()

                # 2. 카메라 프레임 읽기
                wrist_frames = self.wrist_cams.read_all()
                left_rgb = wrist_frames.get("left_wrist")
                right_rgb = wrist_frames.get("right_wrist")

                if left_rgb is None:
                    left_rgb = np.zeros((self.cfg.wrist_height, self.cfg.wrist_width, 3), dtype=np.uint8)
                else:
                    left_rgb = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB)

                if right_rgb is None:
                    right_rgb = np.zeros((self.cfg.wrist_height, self.cfg.wrist_width, 3), dtype=np.uint8)
                else:
                    right_rgb = cv2.cvtColor(right_rgb, cv2.COLOR_BGR2RGB)

                # 3. OS30A 뎁스 및 포인트클라우드
                ret_d, depth_img = self.os30a.read_depth()
                ret_p, pcd_pts = self.os30a.read_pointcloud()

                # 녹화 중이면 버퍼에 누적
                if recording:
                    buffer_qpos.append(follower_qpos)
                    buffer_action.append(leader_action)
                    buffer_left.append(left_rgb)
                    buffer_right.append(right_rgb)
                    buffer_depth.append(depth_img)
                    buffer_pointcloud.append(pcd_pts)
                    sys.stdout.write(f"\r🔴 [녹화 중...] 프레임: {len(buffer_qpos):04d} (FPS: {self.fps})")
                    sys.stdout.flush()

                # 프리뷰 창 표시
                if self.show_preview:
                    preview_left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
                    preview_right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
                    combined = np.hstack([preview_left, preview_right])
                    status_text = "REC" if recording else "STANDBY"
                    color = (0, 0, 255) if recording else (0, 255, 0)
                    cv2.putText(combined, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.imshow("DAPIER-ACT Dual Wrist Preview", combined)
                    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                        break

                # 주기 맞춤
                elapsed = time.time() - loop_start
                sleep_t = max(0.0, self.dt - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        finally:
            self.cleanup()

    def save_episode(self, qpos, action, left, right, depth, pcd):
        if len(qpos) < 10:
            print("⚠️ 에피소드가 너무 짧아(10프레임 미만) 저장하지 않고 폐기합니다.")
            return

        ep_idx = self.get_next_episode_index()
        save_path = self.output_dir / f"episode_{ep_idx}.hdf5"

        with h5py.File(save_path, "w") as f:
            obs_grp = f.create_group("observations")
            img_grp = obs_grp.create_group("images")

            obs_grp.create_dataset("qpos", data=np.array(qpos, dtype=np.float32))
            f.create_dataset("action", data=np.array(action, dtype=np.float32))

            img_grp.create_dataset("left_wrist", data=np.array(left, dtype=np.uint8))
            img_grp.create_dataset("right_wrist", data=np.array(right, dtype=np.uint8))

            obs_grp.create_dataset("depth", data=np.array(depth, dtype=np.uint16))
            obs_grp.create_dataset("pointcloud", data=np.array(pcd, dtype=np.float32))

            f.attrs["task_name"] = self.task_name
            f.attrs["fps"] = self.fps
            f.attrs["total_frames"] = len(qpos)
            f.attrs["episode_index"] = ep_idx

        print(f"✅ [저장 완료] {save_path.resolve()} (프레임 수: {len(qpos)})")

    def cleanup(self):
        print("\n🧹 시스템 자원 정리 중...")
        self.os30a.release()
        self.wrist_cams.stop()
        self.arms.disconnect()
        if self.show_preview:
            cv2.destroyAllWindows()
        print("✨ 정상 종료 완료.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 DAPIER-ACT 멀티모달 데이터 수집기")
    parser.add_argument("--task", type=str, default="dapier_dual_arm_pick", help="작업 이름 (data/ 폴더 하위)")
    parser.add_argument("--fps", type=int, default=20, help="녹화 프레임률")
    parser.add_argument("--left_wrist_cam", type=int, default=4, help="Left Wrist 카메라 V4L2 ID")
    parser.add_argument("--right_wrist_cam", type=int, default=6, help="Right Wrist 카메라 V4L2 ID")
    parser.add_argument("--left_follower", type=str, default="/dev/ttyACM0", help="Left Follower 포트")
    parser.add_argument("--right_follower", type=str, default="/dev/ttyACM1", help="Right Follower 포트")
    parser.add_argument("--left_leader", type=str, default="/dev/ttyACM2", help="Left Leader 포트")
    parser.add_argument("--right_leader", type=str, default="/dev/ttyACM3", help="Right Leader 포트")
    parser.add_argument("--no_preview", action="store_true", help="프리뷰 창 비활성화")
    args = parser.parse_args()

    recorder = DapierEpisodeRecorder(
        task_name=args.task,
        fps=args.fps,
        left_wrist_cam_id=args.left_wrist_cam,
        right_wrist_cam_id=args.right_wrist_cam,
        left_follower_port=args.left_follower,
        right_follower_port=args.right_follower,
        left_leader_port=args.left_leader,
        right_leader_port=args.right_leader,
        show_preview=not args.no_preview,
    )
    recorder.run()


if __name__ == "__main__":
    main()
