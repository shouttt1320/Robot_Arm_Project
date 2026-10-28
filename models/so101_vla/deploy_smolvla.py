"""SO-101 SmolVLA 자연어 지시 기반 실시간 자율 로봇 배포 스크립트.

자연어 명령(예: "Pick up the blue ball and place it in the box" 또는 "파란 공을 상자에 넣어라")을
SmolVLA 멀티모달 정책 모델에 입력하여 실제 SO-101 로봇 팔을 자율 제어합니다.
"""

import argparse
import json
import os
import signal
import sys
import time
import cv2
import numpy as np
import torch

try:
    from hardware.camera_interface import MultiCameraCapture
    from hardware.sts3215_driver import STS3215Driver
    from hardware.identify_arms import find_port_by_serial
except ImportError:
    from camera_interface import MultiCameraCapture
    from sts3215_driver import STS3215Driver
    try:
        from identify_arms import find_port_by_serial
    except ImportError:
        find_port_by_serial = None

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def raw_to_rad(raw_val: int) -> float:
    return (raw_val - 2048) * (2.0 * np.pi / 4096.0)


def rad_to_raw(rad_val: float) -> int:
    return int(2048 + rad_val * (4096.0 / (2.0 * np.pi)))


class SmolVLARobotDeployer:
    """학습된 SmolVLA 정책을 실제 SO-101 팔로워 로봇에 배포하는 추론 컨트롤러."""

    def __init__(
        self,
        checkpoint_path: str = "ckpt/smolvla_so101/checkpoints/last/pretrained_model",
        task: str = "Pick up the blue ball and place it in the box",
        config_path: str = "configs/calibration.json",
        top_cam_id: int = 4,
        wrist_cam_id: int = 6,
        fps: int = 30,
        show_preview: bool = True,
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.task = task
        self.fps = fps
        self.dt = 1.0 / fps
        self.show_preview = show_preview and bool(os.environ.get("DISPLAY"))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.running = False
        self._stopped = False

        # 1. 캘리브레이션 및 하드웨어 설정 로드
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 파일이 없습니다.")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        raw_follower_port = self.config.get("follower_port", "/dev/ttyACM1")
        follower_serial = self.config.get("follower_serial", "")

        if find_port_by_serial and follower_serial:
            self.follower_port = find_port_by_serial(follower_serial, default_port=raw_follower_port)
            if self.follower_port != raw_follower_port:
                print(f"🔍 [포트 자동 매칭] 고유 시리얼({follower_serial}) 기반 팔로워 포트 인식: {self.follower_port}")
        else:
            self.follower_port = raw_follower_port

        self.joints_cfg = self.config["joints"]
        self.gripper_cfg = self.config["gripper"]

        # 2. SmolVLA 정책 및 전처리/후처리 파이프라인 로드
        print(f"\n🧠 [모델 로드] SmolVLA 정책 가중치 로드 중: {checkpoint_path}")
        adapter_cfg_file = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_cfg_file):
            print("📦 [PEFT] LoRA 어댑터 감지됨. 베이스 모델과 어댑터를 함께 로드합니다...")
            from peft import PeftModel
            with open(adapter_cfg_file, "r") as f:
                ad_cfg = json.load(f)
            base_model_path = ad_cfg.get("base_model_name_or_path", "lerobot/smolvla_base")
            base_policy = SmolVLAPolicy.from_pretrained(base_model_path)
            self.policy = PeftModel.from_pretrained(base_policy, checkpoint_path)
        else:
            self.policy = SmolVLAPolicy.from_pretrained(checkpoint_path)

        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=checkpoint_path
        )
        print("✅ SmolVLA 모델 및 입출력 프로세서 준비 완료!")

        # 3. 하드웨어 드라이버 초기화
        self.f_driver = STS3215Driver(self.follower_port, baudrate=1000000)
        self.cam_dict = {"top": top_cam_id, "wrist": wrist_cam_id}
        self.camera_manager = MultiCameraCapture(self.cam_dict, width=640, height=480, fps=self.fps)

        self.default_home_pos = {1: 2196, 2: 912, 3: 2460, 4: 3057, 5: 1844, 6: 1308}
        self.home_pos = None

    def connect(self):
        print("\n" + "=" * 65)
        print(" 🤖 [SO-101 SmolVLA 자연어 지시 자율 구동 (Autonomous Deployment)]")
        print(f" 🦾 대상 로봇: Follower ({self.follower_port})")
        print(f" 🎥 센서    : Top(/dev/video{self.cam_dict['top']}) + Wrist(/dev/video{self.cam_dict['wrist']})")
        print(f" 💬 자연어 명령 : \"{self.task}\"")
        print(f" 🎛️ 제어 주기: {self.fps} Hz | Device: {self.device}")
        print("=" * 65)

        if not self.f_driver.connect():
            raise ConnectionError(f"팔로워 로봇({self.follower_port}) 연결 실패")

        # 팔로워 토크 켜기 전/후 현재 자세를 읽어서 홈 포즈로 지정
        current_follower_pos = self.f_driver.read_positions(range(1, 7))
        if len(current_follower_pos) == 6:
            self.home_pos = current_follower_pos.copy()
            print(f"🏠 홈 포즈 등록 완료 (현재 로봇 시작 위치): {self.home_pos}")
        else:
            self.home_pos = self.default_home_pos.copy()
            print(f"🏠 홈 포즈 등록 완료 (기본 시작 포즈 적용): {self.home_pos}")

        # 팔로워 토크 활성화
        self.f_driver.set_all_torques(range(1, 7), True)
        print("✅ 팔로워 로봇 토크 활성화 완료")

        self.camera_manager.start()
        print("✅ 카메라 캡처 스트림 연결 완료")

    def return_to_home(self, duration: float = 2.5):
        """추론 종료 후 현재 위치에서 홈 포즈로 부드럽게 S-Curve 보간 이동."""
        if self.home_pos is None or not (self.f_driver and self.f_driver.ser and self.f_driver.ser.is_open):
            return

        print("\n🏠 [홈 복귀] 시작 홈 포즈로 부드럽게 복귀합니다...")
        start_pos = self.f_driver.read_positions(range(1, 7))
        if len(start_pos) < 6:
            self.f_driver.sync_write_positions(self.home_pos, speed=0, acc=0)
            time.sleep(duration)
            return

        steps = max(10, int(duration * self.fps))
        dt = duration / steps

        for step in range(1, steps + 1):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (step / steps)))
            interp_goals = {}
            for i in range(1, 7):
                p0 = start_pos.get(i, self.home_pos[i])
                p1 = self.home_pos[i]
                interp_goals[i] = int(round(p0 + (p1 - p0) * alpha))

            self.f_driver.sync_write_positions(interp_goals, speed=0, acc=0)
            time.sleep(dt)

        print("✅ 홈 포즈 복귀 완료.")
        time.sleep(0.3)

    def run(self, max_timesteps: int = 600):
        self.connect()
        self.running = True

        print("\n" + "-" * 65)
        print(" 🟢 [자율 제어 가동 시작!] SmolVLA가 시각 센서와 지시문을 보고 행동합니다.")
        print(f" 💬 Task: \"{self.task}\"")
        print(" ⌨️  (정지하려면 프리뷰 창에서 [Q] 또는 터미널 [Ctrl + C])")
        print("-" * 65 + "\n")

        # 모델 웜업
        print("⚡ [모델 웜업] CUDA 지연 방지를 위해 사전 웜업 추론 수행...")
        with torch.no_grad():
            dummy_obs = {
                "observation.state": torch.zeros(6),
                "observation.images.top": torch.zeros(3, 480, 640, dtype=torch.float32),
                "observation.images.wrist": torch.zeros(3, 480, 640, dtype=torch.float32),
                "task": self.task,
            }
            processed_dummy = self.preprocessor(dummy_obs)
            self.policy.select_action(processed_dummy)
            self.policy.reset()  # 큐 리셋
        print("✅ 모델 GPU 웜업 완료.")

        try:
            for t in range(max_timesteps):
                if not self.running:
                    break
                t0 = time.time()

                # 1. 팔로워 현재 관절값 읽기 (라디안 단위 변환)
                f_pos = self.f_driver.read_positions(range(1, 7))
                curr_qpos = [raw_to_rad(f_pos.get(i, 2048)) for i in range(1, 7)]
                state_tensor = torch.tensor(curr_qpos, dtype=torch.float32)

                # 2. 실시간 카메라 프레임 읽기 (RGB, [0, 1] float, (C, H, W))
                cam_frames = self.camera_manager.read_all()
                top_frame = cam_frames.get("top")
                wrist_frame = cam_frames.get("wrist")

                if top_frame is None:
                    top_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if wrist_frame is None:
                    wrist_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                rgb_top = cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB)
                rgb_wrist = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)

                top_tensor = torch.from_numpy(rgb_top).permute(2, 0, 1).float() / 255.0
                wrist_tensor = torch.from_numpy(rgb_wrist).permute(2, 0, 1).float() / 255.0

                # 3. SmolVLA 전처리 파이프라인 적용
                obs = {
                    "observation.state": state_tensor,
                    "observation.images.top": top_tensor,
                    "observation.images.wrist": wrist_tensor,
                    "task": self.task,
                }
                processed_obs = self.preprocessor(obs)

                # 4. 정책 추론
                t_inf = time.time()
                with torch.no_grad():
                    action_out = self.policy.select_action(processed_obs)
                    action_unnorm = self.postprocessor(action_out)  # (1, 6) 또는 (6,)

                inf_ms = (time.time() - t_inf) * 1000.0
                target_action = action_unnorm.squeeze(0).cpu().numpy()

                # 5. 관절 안전 범위 클램핑 및 하드웨어 목표 각도 전송
                follower_goals = {}
                for i in range(1, 6):
                    cfg = self.joints_cfg[f"joint_{i}"]
                    raw_target = rad_to_raw(target_action[i - 1])
                    min_lim = cfg.get("follower_min", cfg.get("min", 0))
                    max_lim = cfg.get("follower_max", cfg.get("max", 4095))
                    follower_goals[i] = int(clamp(raw_target, min_lim, max_lim))

                # 그리퍼 클램핑
                grip_raw = rad_to_raw(target_action[5])
                f_open = self.gripper_cfg["follower_open"]
                f_close = self.gripper_cfg["follower_close"]
                follower_goals[6] = int(clamp(grip_raw, min(f_open, f_close), max(f_open, f_close)))

                self.f_driver.sync_write_positions(follower_goals, speed=0, acc=0)

                # 6. 실시간 프리뷰 디스플레이 (GUI 불가 시 터미널 진행 로그로 자동 전환)
                if self.show_preview and t % 2 == 0:
                    try:
                        vis_top = top_frame.copy()
                        vis_wrist = wrist_frame.copy()

                        # 정보 오버레이
                        cv2.putText(
                            vis_top,
                            f"Task: {self.task}",
                            (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            vis_top,
                            f"Step: {t+1}/{max_timesteps} | Inf: {inf_ms:.1f}ms",
                            (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                        combined = np.hstack([vis_top, vis_wrist])
                        cv2.imshow("SO-101 SmolVLA Deployment", combined)
                        key = cv2.waitKey(1) & 0xFF
                        if key in [ord("q"), ord("Q")]:
                            print("\n🛑 [사용자 입력] 'Q' 키를 눌러 동작을 중지합니다.")
                            break
                    except Exception as e:
                        print(f"\n⚠️ [화면 프리뷰 비활성화] GUI 미지원 환경 감지: {e}")
                        print("👉 터미널 텍스트 진행 모드로 전환합니다.")
                        self.show_preview = False

                if not self.show_preview and t % 15 == 0:
                    goals_list = [follower_goals[i] for i in range(1, 7)]
                    print(
                        f"\r🚀 [Step {t+1:3d}/{max_timesteps}] Task: \"{self.task}\" | Inf: {inf_ms:.1f}ms | Motors: {goals_list}",
                        end="",
                        flush=True,
                    )

                # 루프 레이트 유지 (30 Hz)
                elapsed = time.time() - t0
                sleep_time = max(0.0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 [인터럽트] Ctrl+C 로 중단되었습니다.")
        finally:
            self.stop()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.running = False

        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        try:
            self.return_to_home(duration=2.5)
        except Exception as e:
            print(f"⚠️ 홈 복귀 중 오류: {e}")

        # 토크 해제
        try:
            if self.f_driver and self.f_driver.ser and self.f_driver.ser.is_open:
                self.f_driver.set_all_torques(range(1, 7), False)
                print("💤 팔로워 모터 토크 해제 완료 (안전 상태)")
        except Exception as e:
            print(f"⚠️ 토크 해제 중 오류: {e}")

        try:
            self.camera_manager.stop()
        except Exception:
            pass
        try:
            self.f_driver.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="SO-101 SmolVLA 자연어 지시 로봇 자율 배포")
    parser.add_argument(
        "--task",
        type=str,
        default="Pick up the blue ball and place it in the box",
        help="로봇에 전달할 자연어 태스크 지시문 (예: 'Pick up the blue ball and place it in the box' 또는 '파란 공 상자에 넣어')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpt/smolvla_so101/checkpoints/last/pretrained_model",
        help="학습된 SmolVLA 체크포인트 디렉터리 경로",
    )
    parser.add_argument("--top-cam", type=int, default=4, help="Top 카메라 인덱스 (/dev/videoN)")
    parser.add_argument("--wrist-cam", type=int, default=6, help="Wrist 카메라 인덱스 (/dev/videoN)")
    parser.add_argument("--max-steps", type=int, default=600, help="최대 추론 스텝 (30fps 기준 600 = 약 20초)")
    parser.add_argument("--no-preview", action="store_true", help="화면 프리뷰 비활성화")
    parser.add_argument("--device", type=str, default="cuda", help="추론 디바이스 (cuda 또는 cpu)")
    args = parser.parse_args()

    deployer = SmolVLARobotDeployer(
        checkpoint_path=args.checkpoint,
        task=args.task,
        top_cam_id=args.top_cam,
        wrist_cam_id=args.wrist_cam,
        show_preview=not args.no_preview,
        device=args.device,
    )

    def sigint_handler(sig, frame):
        print("\n\n[종료 신호 감지] 안전하게 정지 절차를 수행합니다...")
        deployer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    deployer.run(max_timesteps=args.max_steps)


if __name__ == "__main__":
    main()
