"""SO-101 양안(Dual Eyes) DINOv2 ACT 실시간 자율 추론 및 로봇 배포 스크립트."""

import argparse
import csv
import json
import os
import pickle
import signal
import sys
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

try:
    from hardware.camera_interface import MultiCameraCapture
    from hardware.sts3215_driver import STS3215Driver
    from hardware.identify_arms import find_port_by_serial
except ImportError:
    try:
        from camera_interface import MultiCameraCapture
        from sts3215_driver import STS3215Driver
        from identify_arms import find_port_by_serial
    except ImportError:
        find_port_by_serial = None

from so101_dual_dinov2_act.config import (
    BACKBONE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    FIXED_FOCUS_VALUE,
    HIDDEN_DIM,
    IMAGE_SIZE,
    LATENT_DIM,
    NHEADS,
    STATE_DIM,
    TEMPORAL_AGG_M,
)
from so101_dual_dinov2_act.models import build_act_model


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def raw_to_rad(raw_val: int) -> float:
    return (raw_val - 2048) * (2.0 * np.pi / 4096.0)


def rad_to_raw(rad_val: float) -> int:
    return int(round(2048.0 + rad_val * (4096.0 / (2.0 * np.pi))))


class RealRobotDeployer:
    """학습된 양안 DINOv2 ACT 정책을 실제 SO-101 팔로워 로봇에 배포하는 실시간 자율 제어기."""

    def __init__(
        self,
        ckpt_dir: str = CKPT_DIR,
        ckpt_name: str = "policy_best.pth",
        config_path: str = "configs/calibration.json",
        left_cam_id: int = 4,
        right_cam_id: int = 6,
        focus: int = FIXED_FOCUS_VALUE,
        fps: int = 30,
        temporal_agg: bool = True,
        effective_chunk: int = None,
        show_preview: bool = True,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.temporal_agg = temporal_agg
        self.effective_chunk = effective_chunk if effective_chunk is not None else CHUNK_SIZE
        if self.effective_chunk > CHUNK_SIZE:
            self.effective_chunk = CHUNK_SIZE
        self.show_preview = show_preview and bool(os.environ.get("DISPLAY"))
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self.running = False
        self._stopped = False

        # 1. 캘리브레이션 및 한계치 로드
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 파일이 없습니다.")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        raw_follower_port = self.config.get("follower_port", "/dev/ttyACM0")
        follower_serial = self.config.get("follower_serial", "")
        if find_port_by_serial and follower_serial:
            self.follower_port = find_port_by_serial(follower_serial, default_port=raw_follower_port)
            if self.follower_port != raw_follower_port:
                print(f"🔍 [포트 자동 매칭] 고유 시리얼({follower_serial}) 기반 팔로워 포트: {self.follower_port}")
        else:
            self.follower_port = raw_follower_port

        self.joints_cfg = self.config["joints"]
        self.gripper_cfg = self.config["gripper"]

        # 2. 정규화 통계 로드
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"{stats_path} 가 없습니다. 먼저 모델을 학습시키세요.")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        self.qpos_mean = torch.from_numpy(self.stats["qpos_mean"]).float()
        self.qpos_std = torch.from_numpy(self.stats["qpos_std"]).float()
        self.action_mean = torch.from_numpy(self.stats["action_mean"]).float()
        self.action_std = torch.from_numpy(self.stats["action_std"]).float()

        # 3. DINOv2 ACT 모델 빌드 및 가중치 로드
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} 가중치 파일이 없습니다.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ 추론 연산 장치: {self.device}")

        self.model = build_act_model(
            state_dim=STATE_DIM,
            num_queries=CHUNK_SIZE,
            camera_names=CAMERA_NAMES,
            backbone_name=BACKBONE,
            hidden_dim=HIDDEN_DIM,
            dim_feedforward=DIM_FEEDFORWARD,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            nheads=NHEADS,
            latent_dim=LATENT_DIM,
            train_backbone=False,
        )

        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ 양안 DINOv2 가중치 로드 완료 ➔ {ckpt_path}")

        # 4. DINOv2 이미지 전처리 파이프라인 (Bicubic Resize + ImageNet Normalization)
        self.image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 5. 카메라 인터페이스 및 드라이버 설정 (초점 고정)
        self.cam_ids = {CAMERA_NAMES[0]: left_cam_id, CAMERA_NAMES[1]: right_cam_id}
        focus_dict = {name: focus for name in self.cam_ids}
        self.camera_cap = MultiCameraCapture(
            camera_dict=self.cam_ids,
            width=640,
            height=480,
            fps=fps,
            disable_autofocus=True,
            focus_dict=focus_dict,
        )
        self.driver = STS3215Driver(self.follower_port, baudrate=1000000, timeout=0.03)

        # 6. Temporal Ensembling 버퍼
        self.temporal_weights = np.exp(-TEMPORAL_AGG_M * np.arange(self.effective_chunk))
        self.home_pos = None

    def connect(self):
        print("\n" + "=" * 68)
        print(" 🤖 [SO-101 양안(Dual Eyes) DINOv2 ACT 실시간 자율 추론]")
        print(f" 🦾 대상 로봇   : Follower ({self.follower_port})")
        print(f" 👁️ 좌안 카메라 : /dev/video{self.cam_ids[CAMERA_NAMES[0]]}")
        print(f" 👁️ 우안 카메라 : /dev/video{self.cam_ids[CAMERA_NAMES[1]]}")
        print(f" 🧠 모델 백본   : DINOv2 ViT-S/14 (Frozen) + ACT CVAE-Transformer")
        print(f" ⏱️ 제어 주기   : {self.fps} Hz | Temporal Ensembling: {'ON' if self.temporal_agg else 'OFF'}")
        print("=" * 68)

        if not self.driver.connect():
            raise ConnectionError(f"팔로워 로봇({self.follower_port}) 연결 실패!")

        # 시작 전 현재 위치를 읽어 홈 포즈로 저장
        current_pos = self.driver.read_positions(range(1, 7))
        if len(current_pos) == 6:
            self.home_pos = current_pos.copy()
            print(f"🏠 홈 포즈 등록 완료 (현재 로봇 시작 위치): {self.home_pos}")
        else:
            self.home_pos = {
                1: self.joints_cfg["joint_1"]["follower_zero"],
                2: self.joints_cfg["joint_2"]["follower_zero"],
                3: self.joints_cfg["joint_3"]["follower_zero"],
                4: self.joints_cfg["joint_4"]["follower_zero"],
                5: self.joints_cfg["joint_5"]["follower_zero"],
                6: self.gripper_cfg["follower_close"],
            }
            print(f"🏠 홈 포즈 등록 완료 (캘리브레이션 기본 영점): {self.home_pos}")

        self.driver.set_all_torques(range(1, 7), True)
        print("✅ 팔로워 로봇 토크 활성화 완료 (자율 주행 준비)")

        self.camera_cap.start()
        print("✅ 양안 카메라 실시간 캡처 준비 완료")

    def return_to_home(self, duration: float = 2.5):
        """추론 종료 후 현재 위치에서 홈 포즈로 부드럽게 S-Curve 보간 이동."""
        if self.home_pos is None or not (self.driver and self.driver.ser and self.driver.ser.is_open):
            return

        print("\n🏠 [홈 복귀] 추론 완료 후 시작 홈 포즈로 부드럽게 복귀합니다...")
        start_pos = self.driver.read_positions(range(1, 7))
        if len(start_pos) < 6:
            self.driver.sync_write_positions(self.home_pos, speed=0, acc=0)
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

            self.driver.sync_write_positions(interp_goals, speed=0, acc=0)
            time.sleep(dt)

        print("✅ 홈 포즈 복귀 완료.")
        time.sleep(0.3)

    def run(self, max_timesteps: int = 600):
        self.connect()
        self.running = True

        print("\n" + "-" * 68)
        print(" 🟢 [자율 주행 시작] 로봇이 양안 카메라를 보고 자율 동작을 수행합니다.")
        print(f" 📦 [청크 설정] 모델 청크: {CHUNK_SIZE}스텝 | 실효 윈도우: {self.effective_chunk}스텝 ({self.effective_chunk * self.dt:.2f}초)")
        print("    (안전하게 중단하려면 프리뷰 창에서 [Q] 또는 터미널 [Ctrl + C])")
        print("-" * 68 + "\n")

        total_time = max_timesteps + self.effective_chunk
        all_time_actions = torch.zeros([total_time, total_time, STATE_DIM])
        weights = torch.from_numpy(self.temporal_weights).float()

        # 모델 GPU 웜업
        print("⚡ [모델 웜업] 첫 스텝 지연 방지를 위해 사전 웜업을 수행합니다...")
        with torch.inference_mode():
            dummy_qpos = torch.zeros(1, STATE_DIM, device=self.device)
            dummy_img = torch.zeros(1, len(CAMERA_NAMES), 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=self.device)
            self.model(qpos=dummy_qpos, image=dummy_img)
        print("✅ 모델 GPU 웜업 완료.")

        loop_start_time = time.time()
        last_raw_qpos = {}

        try:
            for t in range(max_timesteps):
                if not self.running:
                    break
                t0 = time.time()

                # 1. 팔로워 현재 관절값 읽기 (qpos)
                f_pos = self.driver.read_positions(range(1, 7))
                curr_qpos = []
                for i in range(1, 7):
                    val = f_pos.get(i)
                    if val is None and last_raw_qpos.get(i) is not None:
                        val = last_raw_qpos[i]
                    if val is None:
                        val = 2048
                    last_raw_qpos[i] = val
                    curr_qpos.append(raw_to_rad(val))

                qpos_tensor = torch.from_numpy(np.array(curr_qpos, dtype=np.float32))
                qpos_norm = (qpos_tensor - self.qpos_mean) / self.qpos_std
                qpos_input = qpos_norm.unsqueeze(0).to(self.device)

                # 2. 실시간 양안 카메라 프레임 캡처 및 DINOv2 전처리
                cam_frames = self.camera_cap.read_all()
                cam_tensors = []
                for name in CAMERA_NAMES:
                    frame = cam_frames.get(name)
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    img_t = self.image_transform(img_t)
                    cam_tensors.append(img_t)

                img_input = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(self.device)

                # 3. ACT 모델 추론 (z=0 Prior 모드)
                t_inf_start = time.time()
                with torch.inference_mode():
                    a_hat, _, _ = self.model(qpos=qpos_input, image=img_input)
                inf_ms = (time.time() - t_inf_start) * 1000.0

                # 역정규화
                a_hat_cpu = a_hat.squeeze(0).cpu()
                action_chunk = a_hat_cpu * self.action_std + self.action_mean

                # 4. 액션 결정 (Temporal Aggregation 적용)
                if self.temporal_agg:
                    all_time_actions[[t], t : t + self.effective_chunk] = action_chunk[: self.effective_chunk]
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                    k = len(actions_for_curr_step)
                    step_weights = weights[:k]
                    step_weights = step_weights / step_weights.sum()

                    target_action = (actions_for_curr_step * step_weights.unsqueeze(-1)).sum(axis=0).numpy()
                else:
                    target_action = action_chunk[0].numpy()

                # 5. 관절 안전 한계값 클램핑 및 하드웨어 목표각 전송
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
                grip_min = min(f_open, f_close)
                grip_max = max(f_open, f_close)
                follower_goals[6] = int(clamp(grip_raw, grip_min, grip_max))

                self.driver.sync_write_positions(follower_goals, speed=0, acc=0)

                # 6. 실시간 양안 프리뷰 및 HUD
                elapsed = time.time() - loop_start_time
                if self.show_preview:
                    left_img = cam_frames.get("cam_left")
                    right_img = cam_frames.get("cam_right")
                    target_w, target_h = 480, 360
                    if left_img is not None:
                        left_img = cv2.resize(left_img, (target_w, target_h))
                    else:
                        left_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

                    if right_img is not None:
                        right_img = cv2.resize(right_img, (target_w, target_h))
                    else:
                        right_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

                    # 좌/우안 색상 헤더
                    cv2.rectangle(left_img, (0, 0), (target_w, 28), (40, 160, 40), -1)
                    cv2.putText(left_img, "◀ LEFT EYE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.rectangle(right_img, (0, 0), (target_w, 28), (20, 120, 240), -1)
                    cv2.putText(right_img, "RIGHT EYE ▶", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    combined = np.hstack([left_img, right_img])
                    h, w = combined.shape[:2]
                    cv2.rectangle(combined, (0, h - 38), (w, h), (0, 140, 0), -1)
                    hud = f"DINOv2 ACT [Step {t+1:3d}/{max_timesteps} | {elapsed:4.1f}s | {inf_ms:4.1f}ms] - Press Q to STOP"
                    cv2.putText(combined, hud, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.namedWindow("SO-101 Dual-Eyes DINOv2 ACT [Autonomous]", cv2.WINDOW_NORMAL)
                    cv2.imshow("SO-101 Dual-Eyes DINOv2 ACT [Autonomous]", combined)
                    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                        print("\n⏹️ 사용자가 프리뷰 창에서 Q를 눌러 자율 주행을 중단했습니다.")
                        break

                sys.stdout.write(f"\r  🤖 자율 주행 실행 중... 스텝: {t+1:3d}/{max_timesteps} ({elapsed:4.1f}초, 추론: {inf_ms:4.1f}ms)")
                sys.stdout.flush()

                # 30Hz 제어 주기 유지
                dt_step = time.time() - t0
                sleep_time = self.dt - dt_step
                if sleep_time > 0:
                    time.sleep(sleep_time)

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
            self.return_to_home()
        except Exception as e:
            print(f"[경고] 홈 복귀 중 예외: {e}")

        try:
            # 안전하게 모터 토크 해제
            self.driver.set_all_torques(range(1, 7), False)
            print("💤 팔로워 모터 토크 해제 완료 (대기 모드)")
        except Exception:
            pass

        self.driver.close()
        self.camera_cap.stop()
        print("\n🏁 자율 주행 세션이 정상 종료되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 양안 DINOv2 ACT 실 로봇 자율 배포")
    parser.add_argument("--ckpt", type=str, default="policy_best.pth", help="가중치 파일 이름 (기본값: policy_best.pth)")
    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR, help=f"체크포인트 디렉토리 (기본값: {CKPT_DIR})")
    parser.add_argument("--config", type=str, default="configs/calibration.json", help="캘리브레이션 JSON 경로")
    parser.add_argument("--left-cam", type=int, default=4, help="좌안 카메라 인덱스 (기본값: 4)")
    parser.add_argument("--right-cam", type=int, default=6, help="우안 카메라 인덱스 (기본값: 6)")
    parser.add_argument("--focus", type=int, default=FIXED_FOCUS_VALUE, help="초점 고정 값 (0~255, 기본값: 25)")
    parser.add_argument("--fps", type=int, default=30, help="제어 주기 FPS (기본값: 30)")
    parser.add_argument("--max-steps", type=int, default=600, help="최대 실행 스텝 수 (기본값: 600, 약 20초)")
    parser.add_argument("--no-agg", action="store_true", help="Temporal Aggregation 비활성화 (단일 청크 스텝)")
    parser.add_argument("--chunk", type=int, default=None, help="실효 청크 윈도우 크기 (기본값: 전체 CHUNK_SIZE)")
    parser.add_argument("--no-preview", action="store_true", help="프리뷰 창 끄기 (터미널 모드)")
    args = parser.parse_args()

    deployer = RealRobotDeployer(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt,
        config_path=args.config,
        left_cam_id=args.left_cam,
        right_cam_id=args.right_cam,
        focus=args.focus,
        fps=args.fps,
        temporal_agg=not args.no_agg,
        effective_chunk=args.chunk,
        show_preview=not args.no_preview,
    )

    def sigint_handler(sig, frame):
        deployer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    deployer.run(max_timesteps=args.max_steps)


if __name__ == "__main__":
    main()
