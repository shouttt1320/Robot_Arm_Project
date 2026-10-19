"""SO-101 픽앤플레이스 ACT 실시간 자율 추론 및 로봇 배포 (Phase 6).

학습된 policy_best.ckpt 모델과 2대의 실시간 카메라(Top, Wrist)를 이용해
조작자 개입 없이 로봇 팔이 픽앤플레이스를 완전 자율 수행합니다.
(지수 가중 Temporal Aggregation 적용으로 부드러운 무떨림 궤적 보장)
"""

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
except ImportError:
    from camera_interface import MultiCameraCapture
    from sts3215_driver import STS3215Driver

from so101_act.config import (
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
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


class RealRobotDeployer:
    """학습된 ACT 정책을 실제 SO-101 팔로워 로봇에 배포하는 추론 컨트롤러."""

    def __init__(
        self,
        ckpt_dir: str = CKPT_DIR,
        ckpt_name: str = "policy_best.ckpt",
        config_path: str = "configs/calibration.json",
        top_cam_id: int = 4,
        wrist_cam_id: int = 6,
        fps: int = 30,
        temporal_agg: bool = True,
        effective_chunk: int = None,
        show_preview: bool = True,
        log_diag: bool = True,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.temporal_agg = temporal_agg
        self.effective_chunk = effective_chunk if effective_chunk is not None else CHUNK_SIZE
        if self.effective_chunk > CHUNK_SIZE:
            print(f"[경고] effective_chunk({self.effective_chunk})는 모델 CHUNK_SIZE({CHUNK_SIZE})보다 클 수 없습니다. {CHUNK_SIZE}로 제한합니다.")
            self.effective_chunk = CHUNK_SIZE
        self.show_preview = show_preview and bool(os.environ.get("DISPLAY"))
        self.log_diag = log_diag
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self.running = False
        self._stopped = False

        # 1. 캘리브레이션 및 한계치 로드
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 파일이 없습니다.")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.follower_port = self.config["follower_port"]
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

        # 3. ACT 모델 빌드 및 가중치 로드
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} 가중치 파일이 없습니다.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ 추론 연산 장치: {self.device}")

        self.model = build_act_model(
            state_dim=STATE_DIM,
            num_queries=CHUNK_SIZE,
            camera_names=CAMERA_NAMES,
            hidden_dim=HIDDEN_DIM,
            dim_feedforward=DIM_FEEDFORWARD,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            nheads=NHEADS,
            latent_dim=LATENT_DIM,
        )
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ 학습된 가중치 로드 완료 ➔ {ckpt_path}")

        # 이미지 전처리
        self.normalize_img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 4. 카메라 및 하드웨어 초기화
        self.cam_dict = {"top": top_cam_id, "wrist": wrist_cam_id}
        self.camera_manager = MultiCameraCapture(self.cam_dict, width=640, height=480, fps=fps)
        self.f_driver = STS3215Driver(self.follower_port, baudrate=1000000, timeout=0.03)

        # 5. Temporal Aggregation 버퍼 초기화
        self.chunk_size = CHUNK_SIZE
        # 미래 액션 예측 큐 버퍼
        self.all_time_actions = None

        # 6. 홈 포즈 설정 (학습 데이터 시작 포즈 기본값)
        # Joints 1~6: Base, Shoulder, Elbow, WristPitch, WristRoll, Gripper
        self.default_home_pos = {1: 2196, 2: 912, 3: 2460, 4: 3057, 5: 1844, 6: 1308}
        self.home_pos = None

    def connect(self):
        print("\n" + "=" * 65)
        print(" 🤖 [SO-101 실시간 완전 자율 구동 (Autonomous Deployment)]")
        print(f" 🦾 대상 로봇: Follower ({self.follower_port})")
        print(f" 🎥 센서    : Top(/dev/video{self.cam_dict['top']}) + Wrist(/dev/video{self.cam_dict['wrist']})")
        print(f" 🎛️ 제어 주기: {self.fps} Hz | Temporal Ensembling: {'ON' if self.temporal_agg else 'OFF'}")
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
            print(f"🏠 홈 포즈 등록 완료 (학습 데이터 시작 포즈 적용): {self.home_pos}")

        # 팔로워 토크 활성화
        self.f_driver.set_all_torques(range(1, 7), True)
        print("✅ 팔로워 로봇 토크 활성화 완료 (자율 주행 준비)")

        self.camera_manager.start()
        print("✅ 카메라 캡처 스트림 연결 완료")

    def return_to_home(self, duration: float = 2.5):
        """추론 종료 후 현재 위치에서 홈 포즈로 부드럽게 S-Curve 보간 이동."""
        if self.home_pos is None or not (self.f_driver and self.f_driver.ser and self.f_driver.ser.is_open):
            return

        print("\n\n🏠 [홈 복귀] 추론 완료 후 시작 홈 포즈로 부드럽게 복귀합니다...")
        start_pos = self.f_driver.read_positions(range(1, 7))
        if len(start_pos) < 6:
            print("⚠️ 현재 위치 일부 읽기 실패 ➔ 목표 홈 포즈 직접 전송")
            self.f_driver.sync_write_positions(self.home_pos, speed=0, acc=0)
            time.sleep(duration)
            return

        steps = max(10, int(duration * self.fps))
        dt = duration / steps

        for step in range(1, steps + 1):
            # 코사인 S-Curve 부드러운 가감속 보간 (0.0 ~ 1.0)
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
        print(" 🟢 [자율 제어 루프 가동 시작!] 로봇이 카메라를 보고 물체를 이동시킵니다.")
        print(f" 📦 [청크 설정] 모델 청크: {CHUNK_SIZE}스텝 | 실효 제어 윈도우(Effective Chunk): {self.effective_chunk}스텝 ({self.effective_chunk * self.dt:.2f}초)")
        print("    (안전하게 정지하려면 프리뷰 창에서 [Q] 또는 터미널 [Ctrl + C])")
        print("-" * 65 + "\n")

        # Temporal Ensembling 버퍼 (max_timesteps + effective_chunk)
        total_time = max_timesteps + self.effective_chunk
        all_time_actions = torch.zeros([total_time, total_time, STATE_DIM])

        # 지수 가중치 계산 (k번째 이전 스텝의 예측치에 exp(-m * k) 가중)
        weights = np.exp(-TEMPORAL_AGG_M * np.arange(self.effective_chunk))
        weights = torch.from_numpy(weights).float()

        # 모델 GPU 웜업 (Step 0의 CUDA Cold-start 지연 방지)
        print("⚡ [모델 웜업] 첫 스텝 CUDA 지연 방지를 위해 사전 웜업을 수행합니다...")
        with torch.inference_mode():
            dummy_qpos = torch.zeros(1, STATE_DIM, device=self.device)
            dummy_img = torch.zeros(1, 2, 3, 480, 640, device=self.device)
            self.model(qpos=dummy_qpos, image=dummy_img)
        print("✅ 모델 GPU 웜업 완료.")

        loop_start_time = time.time()
        diag_records = {
            "timestamp": [],
            "loop_dt_ms": [],
            "inference_ms": [],
            "qpos_in_deg": [],
            "qpos_in_raw": [],
            "action_raw_deg": [],
            "action_raw_norm": [],
            "action_agg_deg": [],
            "action_sent_deg": [],
            "action_sent_raw": [],
            "is_clamped": [],
        }
        visual_keyframes = []

        try:
            for t in range(max_timesteps):
                if not self.running:
                    break
                t0 = time.time()

                # 1. 팔로워 현재 관절값 읽기 (qpos)
                f_pos = self.f_driver.read_positions(range(1, 7))
                curr_qpos = [raw_to_rad(f_pos.get(i, 2048)) for i in range(1, 7)]
                qpos_tensor = torch.from_numpy(np.array(curr_qpos, dtype=np.float32))

                # 정규화
                qpos_norm = (qpos_tensor - self.qpos_mean) / self.qpos_std
                qpos_input = qpos_norm.unsqueeze(0).to(self.device)

                # 2. 실시간 카메라 프레임 캡처
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

                # 3. ACT 모델 추론 (z=0 Prior 모드) 및 추론 시간 측정
                t_inf_start = time.time()
                with torch.inference_mode():
                    a_hat, _, _ = self.model(qpos=qpos_input, image=img_input)  # (1, chunk_size, 6)
                inf_ms = (time.time() - t_inf_start) * 1000.0

                # 역정규화
                a_hat_cpu = a_hat.squeeze(0).cpu()  # (chunk_size, 6)
                action_chunk = a_hat_cpu * self.action_std + self.action_mean  # 라디안 단위 복원
                action_raw_rad = action_chunk[0].numpy()  # 순수 0번 액션

                # 4. 액션 결정 (Temporal Aggregation 적용 여부)
                if self.temporal_agg:
                    all_time_actions[[t], t:t + self.effective_chunk] = action_chunk[:self.effective_chunk]
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
                is_clamped_step = []
                for i in range(1, 6):
                    cfg = self.joints_cfg[f"joint_{i}"]
                    raw_target = rad_to_raw(target_action[i - 1])
                    min_lim = cfg.get("follower_min", cfg.get("min", 0))
                    max_lim = cfg.get("follower_max", cfg.get("max", 4095))
                    clamped_val = int(clamp(raw_target, min_lim, max_lim))
                    follower_goals[i] = clamped_val
                    is_clamped_step.append(raw_target != clamped_val)

                # 그리퍼 클램핑
                grip_raw = rad_to_raw(target_action[5])
                f_open = self.gripper_cfg["follower_open"]
                f_close = self.gripper_cfg["follower_close"]
                grip_min = min(f_open, f_close)
                grip_max = max(f_open, f_close)
                clamped_grip = int(clamp(grip_raw, grip_min, grip_max))
                follower_goals[6] = clamped_grip
                is_clamped_step.append(grip_raw != clamped_grip)

                self.f_driver.sync_write_positions(follower_goals, speed=0, acc=0)

                # 진단 로그 기록
                if self.log_diag:
                    action_sent_rad = [raw_to_rad(follower_goals[i]) for i in range(1, 7)]
                    diag_records["timestamp"].append(t0)
                    diag_records["loop_dt_ms"].append((time.time() - t0) * 1000.0)
                    diag_records["inference_ms"].append(inf_ms)
                    diag_records["qpos_in_deg"].append(np.rad2deg(curr_qpos))
                    diag_records["qpos_in_raw"].append([f_pos.get(i, 2048) for i in range(1, 7)])
                    diag_records["action_raw_deg"].append(np.rad2deg(action_raw_rad))
                    diag_records["action_raw_norm"].append(a_hat_cpu[0].numpy())
                    diag_records["action_agg_deg"].append(np.rad2deg(target_action))
                    diag_records["action_sent_deg"].append(np.rad2deg(action_sent_rad))
                    diag_records["action_sent_raw"].append([follower_goals[i] for i in range(1, 7)])
                    diag_records["is_clamped"].append(is_clamped_step)

                # 시각적 진단 키프레임 수집 (10스텝 단위 또는 마지막 스텝)
                if self.log_diag and (t % 10 == 0 or t == max_timesteps - 1):
                    visual_keyframes.append({
                        "step": t + 1,
                        "top": cam_frames.get("top").copy() if cam_frames.get("top") is not None else None,
                        "wrist": cam_frames.get("wrist").copy() if cam_frames.get("wrist") is not None else None,
                        "qpos_deg": np.rad2deg(curr_qpos),
                        "target_deg": np.rad2deg(target_action),
                    })

                # 6. 실시간 프리뷰 및 HUD
                elapsed = time.time() - loop_start_time
                if self.show_preview and cam_frames.get("top") is not None:
                    top_img = cam_frames["top"].copy()
                    wrist_img = cam_frames["wrist"].copy() if cam_frames.get("wrist") is not None else top_img
                    combined = np.hstack([top_img, wrist_img])
                    h, w = combined.shape[:2]
                    cv2.rectangle(combined, (0, h - 45), (w, h), (0, 160, 0), -1)
                    hud = f"AUTONOMOUS ACT [Step {t+1:3d}/{max_timesteps} | {elapsed:4.1f}s] - Press Q to STOP"
                    cv2.putText(combined, hud, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("SO-101 Autonomous Agent [ACT Policy]", combined)
                    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                        print("\n⏹️ 사용자가 프리뷰 창에서 Q를 눌러 자율 주행을 중단했습니다.")
                        break

                sys.stdout.write(f"\r  🤖 자율 주행 실행 중... 스텝: {t+1:3d}/{max_timesteps} ({elapsed:4.1f}초)")
                sys.stdout.flush()

                # 30Hz 주기 유지
                dt_step = time.time() - t0
                sleep_time = self.dt - dt_step
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            if self.log_diag and len(diag_records["timestamp"]) > 5:
                self._save_and_report_diagnostics(diag_records, visual_keyframes)
            self.stop()

    def _save_and_report_diagnostics(self, diag_records: dict, visual_keyframes: list = None):
        """실시간 진단 로그를 .npz 파일로 저장하고 자동 분석 리포트 및 시각 타임라인 생성"""
        try:
            os.makedirs("logs", exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/deploy_diag_{timestamp_str}.npz"
            latest_file = "logs/latest_deploy_diag.npz"

            data_to_save = {k: np.array(v) for k, v in diag_records.items()}
            data_to_save["meta_ckpt_dir"] = np.array(str(self.ckpt_dir))
            data_to_save["meta_temporal_agg_m"] = np.array(float(TEMPORAL_AGG_M))
            data_to_save["meta_effective_chunk"] = np.array(int(self.effective_chunk))
            np.savez_compressed(log_file, **data_to_save)
            np.savez_compressed(latest_file, **data_to_save)

            print(f"\n💾 [진단 로그 저장 완료] ➔ {log_file} ({len(data_to_save['timestamp'])}스텝)")

            # 1. 자동 정밀 진단 리포트 생성 (최신 1회용 + 타임스탬프 영구 보존용)
            history_jitter_png = f"logs/deploy_jitter_{timestamp_str}.png"
            try:
                from collection.diagnose_jitter import analyze_and_plot_diagnostics
                analyze_and_plot_diagnostics(
                    npz_path=latest_file,
                    output_png="logs/deploy_jitter_diagnosis.png",
                    history_png=history_jitter_png
                )
            except Exception as diag_err:
                print(f"[경고] 진단 리포트 생성 중 예외: {diag_err}")

            # 2. 시각적 타임라인 및 스텝별 고화질 사진 생성 (최신 1회용 + 타임스탬프 영구 보존용)
            history_visual_png = f"logs/deploy_visual_{timestamp_str}.png"
            if visual_keyframes:
                try:
                    from collection.diagnose_jitter import save_visual_trajectory_summary
                    save_visual_trajectory_summary(
                        keyframes=visual_keyframes,
                        output_png="logs/latest_deploy_visual.png",
                        history_png=history_visual_png,
                        frames_dir="logs/latest_frames"
                    )
                except Exception as vis_err:
                    print(f"[경고] 시각적 타임라인 생성 중 예외: {vis_err}")

            # 3. 전체 추론 실험 누적 CSV 기록 (A/B 테스트 비교용)
            try:
                csv_summary_path = "logs/all_deploy_runs.csv"
                write_header = not os.path.exists(csv_summary_path)
                with open(csv_summary_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["timestamp", "ckpt_dir", "steps", "mean_loop_ms", "max_loop_ms", "temporal_agg_m", "effective_chunk", "diag_npz", "visual_png", "jitter_png"])
                    writer.writerow([
                        timestamp_str,
                        self.ckpt_dir,
                        len(data_to_save["timestamp"]),
                        round(float(np.mean(data_to_save["loop_dt_ms"])), 2),
                        round(float(np.max(data_to_save["loop_dt_ms"])), 2),
                        float(TEMPORAL_AGG_M),
                        int(self.effective_chunk),
                        log_file,
                        history_visual_png,
                        history_jitter_png
                    ])
                print(f"📑 [추론 실험 히스토리 등록 완료] ➔ {csv_summary_path}")
            except Exception as csv_err:
                print(f"[경고] 추론 누적 CSV 기록 중 예외: {csv_err}")

        except Exception as e:
            print(f"⚠️ 진단 로그 저장 중 오류 발생: {e}")

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

        # 1. 홈 포즈로 부드럽게 복귀
        try:
            self.return_to_home(duration=2.5)
        except Exception as e:
            print(f"⚠️ 홈 복귀 중 예외 발생: {e}")

        # 2. 토크 안전 해제
        print("\n🔓 [토크 해제] 로봇 관절 모터의 토크를 안전하게 해제합니다...")
        try:
            if self.f_driver and self.f_driver.ser and self.f_driver.ser.is_open:
                # 브로드캐스트 해제 + 모터별 개별 해제
                self.f_driver.set_torque(0xFE, False)
                for mid in range(1, 7):
                    self.f_driver.set_torque(mid, False)
                    time.sleep(0.005)
                if hasattr(self.f_driver.ser, "flush"):
                    self.f_driver.ser.flush()
            time.sleep(0.1)
        except Exception as e:
            print(f"[경고] 토크 해제 중 예외 발생: {e}")
        finally:
            if self.f_driver:
                self.f_driver.close()

        try:
            self.camera_manager.stop()
        except Exception:
            pass

        print("✅ 로봇 안전 정지 및 배포 종료 완료.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 실시간 ACT 자율 주행 배포기 (Phase 6)")
    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR, help="체크포인트 폴더 경로")
    parser.add_argument("--ckpt-name", type=str, default="policy_best.ckpt", help="불러올 가중치 파일명 (기본값: policy_best.ckpt, 예: policy_epoch_5000.ckpt)")
    parser.add_argument("--top-cam", type=int, default=4, help="상단 카메라 인덱스")
    parser.add_argument("--wrist-cam", type=int, default=6, help="손목 카메라 인덱스")
    parser.add_argument("--fps", type=int, default=30, help="제어 주기 (기본값: 30Hz)")
    parser.add_argument("--max-steps", type=int, default=600, help="1회 자율 수행 최대 스텝 (기본값: 600스텝/20초)")
    parser.add_argument("--effective-chunk", type=int, default=None, help=f"추론 시 실효 청크 윈도우 크기 (기본값: 모델 청크 {CHUNK_SIZE}, 예: 20으로 줄이면 빠른 실시간 반응)")
    parser.add_argument("--no-agg", action="store_true", help="Temporal Aggregation 비활성화")
    parser.add_argument("--no-preview", action="store_true", help="카메라 HUD 프리뷰 창 비활성화")
    parser.add_argument("--no-log-diag", action="store_true", help="진단 로깅 비활성화")
    args = parser.parse_args()

    deployer = RealRobotDeployer(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        top_cam_id=args.top_cam,
        wrist_cam_id=args.wrist_cam,
        fps=args.fps,
        temporal_agg=not args.no_agg,
        effective_chunk=args.effective_chunk,
        show_preview=not args.no_preview,
        log_diag=not args.no_log_diag,
    )

    def sigint_handler(sig, frame):
        deployer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    deployer.run(max_timesteps=args.max_steps)


if __name__ == "__main__":
    main()
