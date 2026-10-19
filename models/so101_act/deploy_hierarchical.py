"""SO-101 3단계 계층적 모듈(Reach ➔ Grasp ➔ Place) 실시간 자율 추론 컨트롤러.

3개의 전용 서브태스크 모델을 FSM(상태 머신)으로 순차 전환하여,
단일 통짜 모델의 인과 혼동을 방지하고 목표물 파지 및 배치 성공률을 극대화합니다.
추론 완료 또는 정지 시 부드러운 S-Curve 홈 복귀 및 안전 토크 해제를 수행합니다.
"""

import argparse
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
except ImportError:
    from camera_interface import MultiCameraCapture
    from sts3215_driver import STS3215Driver

from so101_act.config import (
    CAMERA_NAMES,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    HIDDEN_DIM,
    LATENT_DIM,
    NHEADS,
    STATE_DIM,
    TEMPORAL_AGG_M,
)
from so101_act.models import build_act_model


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def raw_to_rad(raw_val: int) -> float:
    return (raw_val - 2048) * (2.0 * np.pi / 4096.0)


def rad_to_raw(rad_val: float) -> int:
    return int(2048 + rad_val * (4096.0 / (2.0 * np.pi)))


class SubtaskPolicy:
    """개별 서브태스크(Reach, Grasp, Place)의 가중치 및 정규화 통계 래퍼."""

    def __init__(self, name: str, ckpt_dir: str, chunk_size: int, device: torch.device):
        self.name = name
        self.chunk_size = chunk_size
        self.device = device

        ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[{name}] 가중치 파일이 없습니다: {ckpt_path}")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"[{name}] 통계 파일이 없습니다: {stats_path}")

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        self.qpos_mean = torch.from_numpy(stats["qpos_mean"]).float().to(device)
        self.qpos_std = torch.from_numpy(stats["qpos_std"]).float().to(device)
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]

        self.model = build_act_model(
            state_dim=STATE_DIM,
            num_queries=chunk_size,
            camera_names=CAMERA_NAMES,
            hidden_dim=HIDDEN_DIM,
            dim_feedforward=DIM_FEEDFORWARD,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            nheads=NHEADS,
            latent_dim=LATENT_DIM,
        )
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, qpos_raw: np.ndarray, img_tensors: torch.Tensor) -> np.ndarray:
        qpos_t = torch.from_numpy(qpos_raw).float().to(self.device)
        qpos_norm = (qpos_t - self.qpos_mean) / self.qpos_std
        qpos_input = qpos_norm.unsqueeze(0)

        with torch.inference_mode():
            a_hat, _, _ = self.model(qpos=qpos_input, image=img_tensors)

        a_hat_cpu = a_hat.squeeze(0).cpu().numpy()
        action_chunk = a_hat_cpu * self.action_std + self.action_mean
        return action_chunk


class HierarchicalDeployer:
    """3단계 서브태스크 FSM 제어기."""

    def __init__(
        self,
        subtasks_root: str = "ckpt/so101_subtasks",
        config_path: str = "configs/calibration.json",
        top_cam_id: int = 4,
        wrist_cam_id: int = 6,
        fps: int = 30,
        show_preview: bool = True,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.show_preview = show_preview and bool(os.environ.get("DISPLAY"))
        self.running = False
        self._stopped = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 캘리브레이션 로드
        if not os.path.exists(config_path):
            alt_path = "calibration.json"
            if os.path.exists(alt_path):
                config_path = alt_path
            else:
                raise FileNotFoundError(f"{config_path} 파일이 없습니다.")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.follower_port = self.config["follower_port"]
        self.joints_cfg = self.config["joints"]
        self.gripper_cfg = self.config["gripper"]

        # 2. 3개 서브태스크 모델 로드
        print(f"\n⚙️  추론 디바이스: {self.device}")
        print("📦 3단계 서브태스크 정책 모델 로딩 중...")
        self.policies = {
            "reach": SubtaskPolicy("REACH", os.path.join(subtasks_root, "reach"), chunk_size=50, device=self.device),
            "grasp": SubtaskPolicy("GRASP", os.path.join(subtasks_root, "grasp"), chunk_size=25, device=self.device),
            "place": SubtaskPolicy("PLACE", os.path.join(subtasks_root, "place"), chunk_size=40, device=self.device),
        }
        print("✅ Reach, Grasp, Place 3개 모델 로드 완료!")

        # 이미지 전처리
        self.normalize_img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 3. 하드웨어 드라이버 초기화
        self.cam_dict = {"top": top_cam_id, "wrist": wrist_cam_id}
        self.camera_manager = MultiCameraCapture(self.cam_dict, width=640, height=480, fps=fps)
        self.f_driver = STS3215Driver(self.follower_port, baudrate=1000000, timeout=0.03)

        self.home_pos = None
        self.default_home_pos = {1: 2196, 2: 912, 3: 2460, 4: 3057, 5: 1844, 6: 1308}

    def connect(self):
        print("\n" + "=" * 65)
        print(" 🤖 [SO-101 3단계 계층적 자율 추론 (Reach ➔ Grasp ➔ Place)]")
        print(f" 🦾 대상 로봇: Follower ({self.follower_port})")
        print(f" 🎥 센서    : Top(/dev/video{self.cam_dict['top']}) + Wrist(/dev/video{self.cam_dict['wrist']})")
        print("=" * 65)

        if not self.f_driver.connect():
            raise ConnectionError(f"팔로워 로봇({self.follower_port}) 연결 실패")

        curr = self.f_driver.read_positions(range(1, 7))
        if len(curr) == 6:
            self.home_pos = curr.copy()
            print(f"🏠 홈 포즈 등록 완료 (현재 로봇 시작 위치): {self.home_pos}")
        else:
            self.home_pos = self.default_home_pos.copy()
            print(f"🏠 홈 포즈 기본값 적용: {self.home_pos}")

        self.f_driver.set_all_torques(range(1, 7), True)
        self.camera_manager.start()
        print("✅ 하드웨어 및 카메라 준비 완료.")

    def return_to_home(self, duration: float = 2.5):
        if self.home_pos is None or not (self.f_driver and self.f_driver.ser and self.f_driver.ser.is_open):
            return

        print("\n\n🏠 [홈 복귀] 홈 포즈로 부드럽게 복귀합니다...")
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

        print("✅ 홈 포즈 도착 완료.")
        time.sleep(0.3)

    def execute_subtask(self, stage_name: str, policy: SubtaskPolicy, max_steps: int, exit_condition_fn=None):
        """단일 서브태스크를 지정된 스텝 동안 자율 실행."""
        print(f"\n▶️  [Stage: {stage_name}] 실행 시작 (최대 {max_steps}스텝)")

        for step in range(max_steps):
            if not self.running:
                break
            t0 = time.time()

            # 1. qpos 읽기
            f_pos = self.f_driver.read_positions(range(1, 7))
            curr_qpos = np.array([raw_to_rad(f_pos.get(i, 2048)) for i in range(1, 7)], dtype=np.float32)

            # 조기 전이 조건 확인
            if exit_condition_fn and exit_condition_fn(step, curr_qpos, f_pos):
                print(f"  🏁 [Stage: {stage_name}] 목표 달성 조건 충족 ➔ 다음 단계 전환 (스텝 {step+1})")
                break

            # 2. 이미지 읽기
            cam_frames = self.camera_manager.read_all()
            cam_tensors = []
            for name in ["top", "wrist"]:
                frame = cam_frames.get(name)
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                img_t = self.normalize_img(img_t)
                cam_tensors.append(img_t)

            img_input = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(self.device)

            # 3. 모델 추론
            action_chunk = policy.predict(curr_qpos, img_input)
            target_action = action_chunk[0]  # 즉각 피드백 반응

            # 4. 관절각 클램핑 및 전송
            follower_goals = {}
            for i in range(1, 6):
                cfg = self.joints_cfg[f"joint_{i}"]
                raw_target = rad_to_raw(target_action[i - 1])
                min_lim = cfg.get("follower_min", cfg.get("min", 0))
                max_lim = cfg.get("follower_max", cfg.get("max", 4095))
                follower_goals[i] = int(clamp(raw_target, min_lim, max_lim))

            grip_raw = rad_to_raw(target_action[5])
            f_open = self.gripper_cfg["follower_open"]
            f_close = self.gripper_cfg["follower_close"]
            follower_goals[6] = int(clamp(grip_raw, min(f_open, f_close), max(f_open, f_close)))

            self.f_driver.sync_write_positions(follower_goals, speed=0, acc=0)

            # 5. 프리뷰 HUD
            if self.show_preview and cam_frames.get("top") is not None:
                top_img = cam_frames["top"].copy()
                wrist_img = cam_frames["wrist"].copy() if cam_frames.get("wrist") is not None else top_img
                combined = np.hstack([top_img, wrist_img])
                h, w = combined.shape[:2]
                cv2.rectangle(combined, (0, h - 45), (w, h), (0, 120, 200), -1)
                hud = f"HIERARCHICAL [{stage_name}] Step {step+1:3d}/{max_steps} - Press Q to STOP"
                cv2.putText(combined, hud, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("SO-101 Hierarchical ACT", combined)
                if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                    self.running = False
                    break

            dt_step = time.time() - t0
            sleep_time = self.dt - dt_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run(self):
        self.connect()
        self.running = True

        try:
            # Stage 1: REACH (공 상단 접근)
            def reach_exit(step, qpos, raw_dict):
                # 90스텝 이상 경과하고 그리퍼가 충분히 열려있을 때
                return step > 100

            self.execute_subtask("1. REACH (접근)", self.policies["reach"], max_steps=140, exit_condition_fn=reach_exit)

            # Stage 2: GRASP (파지 및 리프트)
            def grasp_exit(step, qpos, raw_dict):
                # 그리퍼가 닫히고(raw < 1900 or rad < -0.7) 30스텝 이상 경과 시
                return step > 35 and qpos[5] < -0.65

            if self.running:
                self.execute_subtask("2. GRASP (잡기)", self.policies["grasp"], max_steps=60, exit_condition_fn=grasp_exit)

            # Stage 3: PLACE (바구니 이동 및 놓기)
            if self.running:
                self.execute_subtask("3. PLACE (놓기)", self.policies["place"], max_steps=180)

            print("\n🎉 [전체 3단계 픽앤플레이스 자율 완료!]")

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
            print(f"⚠️ 홈 복귀 중 예외: {e}")

        print("\n🔓 [토크 해제] 안전 정지 중...")
        try:
            if self.f_driver and self.f_driver.ser and self.f_driver.ser.is_open:
                self.f_driver.set_torque(0xFE, False)
                for mid in range(1, 7):
                    self.f_driver.set_torque(mid, False)
                    time.sleep(0.005)
                if hasattr(self.f_driver.ser, "flush"):
                    self.f_driver.ser.flush()
        except Exception as e:
            print(f"토크 해제 예외: {e}")
        finally:
            if self.f_driver:
                self.f_driver.close()

        try:
            self.camera_manager.stop()
        except Exception:
            pass

        print("✅ 자율 주행 종료.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 3단계 계층적 자율 주행 배포기")
    parser.add_argument("--subtasks-root", type=str, default="ckpt/so101_subtasks", help="3개 서브태스크 가중치 폴더 루트")
    parser.add_argument("--top-cam", type=int, default=4, help="상단 카메라 인덱스")
    parser.add_argument("--wrist-cam", type=int, default=6, help="손목 카메라 인덱스")
    parser.add_argument("--fps", type=int, default=30, help="제어 주기")
    parser.add_argument("--no-preview", action="store_true", help="프리뷰 비활성화")
    args = parser.parse_args()

    deployer = HierarchicalDeployer(
        subtasks_root=args.subtasks_root,
        top_cam_id=args.top_cam,
        wrist_cam_id=args.wrist_cam,
        fps=args.fps,
        show_preview=not args.no_preview,
    )

    def sigint_handler(sig, frame):
        deployer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    deployer.run()


if __name__ == "__main__":
    main()
