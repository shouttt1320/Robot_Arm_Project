"""SO-101 DAPIER-ACT 실물 양팔 로봇 실시간 배포 및 제어기.

양팔 12 DOF + OS30A 뎁스 포인트클라우드 + Dual Wrist RGB 멀티모달 추론 및 Temporal Ensembling.
"""

import argparse
import os
from pathlib import Path
import pickle
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from hardware.camera_interface import MultiCameraCapture
from hardware.os30a_camera import OS30ACamera
from hardware.dual_arm_controller import DualArmController
from .config import DapierACTConfig
from .model import DapierACTMultimodal


class DapierACTDeployer:
    def __init__(
        self,
        checkpoint_path: str,
        stats_path: Optional[str] = None,
        left_wrist_cam_id: int = 4,
        right_wrist_cam_id: int = 6,
        left_follower_port: str = "/dev/ttyACM0",
        right_follower_port: str = "/dev/ttyACM1",
        fps: int = 20,
        temporal_ensemble: bool = True,
        ensemble_weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.fps = fps
        self.dt = 1.0 / fps
        self.temporal_ensemble = temporal_ensemble
        self.ensemble_weight_decay = ensemble_weight_decay

        print("=" * 70)
        print(" 🧠 [SO-101 DAPIER-ACT 실시간 배포 추론기]")
        print(f" • 체크포인트 : {checkpoint_path}")
        print(f" • 디바이스   : {self.device}")
        print(f" • 제어 주기  : {self.fps} Hz (dt={self.dt*1000:.1f}ms)")
        print("=" * 70)

        # 1. 체크포인트 및 통계 로드
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.cfg = ckpt.get("config", DapierACTConfig())

        # 통계 복원
        if stats_path and Path(stats_path).exists():
            with open(stats_path, "rb") as f:
                self.stats = pickle.load(f)
        elif "stats" in ckpt:
            self.stats = ckpt["stats"]
        else:
            default_stats_path = ckpt_path.parent / "stats.pkl"
            if default_stats_path.exists():
                with open(default_stats_path, "rb") as f:
                    self.stats = pickle.load(f)
            else:
                raise FileNotFoundError(f"정규화 통계 파일(stats.pkl)을 찾을 수 없습니다.")

        # 2. 모델 인스턴스화
        self.model = DapierACTMultimodal(config=self.cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.point_encoder.set_point_limits(self.stats["point_min"], self.stats["point_max"])
        self.model.to(self.device)
        self.model.eval()
        print("✅ DAPIER-ACT 신경망 및 파라미터 로드 완료.")

        # 3. 하드웨어 초기화
        print("\n📷 OS30A 뎁스 카메라 초기화 중...")
        self.os30a = OS30ACamera(config=self.cfg, use_sample_fallback=True)

        print("📷 손목 RGB 카메라 초기화 중...")
        self.wrist_cams = MultiCameraCapture(
            {"left_wrist": left_wrist_cam_id, "right_wrist": right_wrist_cam_id},
            width=self.cfg.wrist_width,
            height=self.cfg.wrist_height,
            fps=fps,
        )

        print("🦾 양팔 SO-101 (12 DOF) 로봇 연결 중...")
        self.arms = DualArmController(
            left_follower_port=left_follower_port,
            right_follower_port=right_follower_port,
        )
        self.arms.connect()

        # Temporal Ensembling 버퍼: 각 미래 시점 t에 대한 예측 목록 저장
        self.all_time_actions = []

    def run(self, max_timesteps: int = 600):
        print("\n🚀 [실시간 자율 주행 시작] (Ctrl+C를 누르면 안전하게 정지합니다)")
        chunk_size = self.cfg.chunk_size

        try:
            for step in range(max_timesteps):
                loop_start = time.time()

                # 1. 하드웨어 관측값 취득
                current_qpos = self.arms.read_follower_qpos()
                norm_qpos = (current_qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
                qpos_t = torch.from_numpy(norm_qpos).unsqueeze(0).to(self.device)

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

                left_t = (
                    torch.from_numpy(left_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                )
                right_t = (
                    torch.from_numpy(right_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                )

                ret_p, pts = self.os30a.read_pointcloud()
                pts_t = torch.from_numpy(pts).unsqueeze(0).to(self.device)

                # 2. DAPIER-ACT 모델 순전파 (미래 20스텝 청크 예측)
                with torch.no_grad():
                    pred_chunk, _, _ = self.model(
                        qpos=qpos_t,
                        pointcloud=pts_t,
                        left_wrist=left_t,
                        right_wrist=right_t,
                    )
                pred_np = pred_chunk.squeeze(0).cpu().numpy()  # [20, 12]
                unnorm_chunk = pred_np * self.stats["action_std"] + self.stats["action_mean"]

                # 3. Temporal Ensembling (가중 지수 평활화)
                if self.temporal_ensemble:
                    self.all_time_actions.append(unnorm_chunk)
                    # 현재 step 시점에 대한 이전 청크들의 예측값 수집
                    actions_for_curr_step = []
                    weights = []
                    for t, chunk in enumerate(self.all_time_actions):
                        k = step - t
                        if 0 <= k < chunk_size:
                            actions_for_curr_step.append(chunk[k])
                            weights.append(np.exp(-self.ensemble_weight_decay * k))

                    weights = np.array(weights)
                    weights = weights / np.sum(weights)
                    target_action = np.sum([w * a for w, a in zip(weights, actions_for_curr_step)], axis=0)
                else:
                    target_action = unnorm_chunk[0]

                # 4. 로봇 팔에 12개 관절 제어 명령 전송
                self.arms.send_action(target_action)

                # 상태 출력
                if step % 10 == 0:
                    left_gripper = target_action[5]
                    right_gripper = target_action[11]
                    print(
                        f"Step: {step:03d} | "
                        f"Left Gripper: {left_gripper:.2f} rad | "
                        f"Right Gripper: {right_gripper:.2f} rad"
                    )

                # 주기 맞춤
                elapsed = time.time() - loop_start
                sleep_t = max(0.0, self.dt - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 자율 주행이 정지되었습니다.")
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n🧹 시스템 자원 해제 중...")
        self.os30a.release()
        self.wrist_cams.stop()
        self.arms.disconnect()
        print("✨ 정상 종료 완료.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 DAPIER-ACT 실시간 배포기")
    parser.add_argument("--checkpoint", type=str, required=True, help="학습된 best_model.pt 파일 경로")
    parser.add_argument("--stats", type=str, default=None, help="정규화 stats.pkl 경로 (생략 시 체크포인트 폴더 내 자동 탐색)")
    parser.add_argument("--fps", type=int, default=20, help="제어 주기")
    parser.add_argument("--left_wrist_cam", type=int, default=4, help="Left Wrist 카메라 ID")
    parser.add_argument("--right_wrist_cam", type=int, default=6, help="Right Wrist 카메라 ID")
    parser.add_argument("--left_follower", type=str, default="/dev/ttyACM0", help="Left Follower 포트")
    parser.add_argument("--right_follower", type=str, default="/dev/ttyACM1", help="Right Follower 포트")
    parser.add_argument("--no_ensemble", action="store_true", help="Temporal Ensembling 비활성화")
    parser.add_argument("--max_steps", type=int, default=600, help="최대 실행 스텝 수")
    args = parser.parse_args()

    deployer = DapierACTDeployer(
        checkpoint_path=args.checkpoint,
        stats_path=args.stats,
        left_wrist_cam_id=args.left_wrist_cam,
        right_wrist_cam_id=args.right_wrist_cam,
        left_follower_port=args.left_follower,
        right_follower_port=args.right_follower,
        fps=args.fps,
        temporal_ensemble=not args.no_ensemble,
    )
    deployer.run(max_timesteps=args.max_steps)


if __name__ == "__main__":
    main()
