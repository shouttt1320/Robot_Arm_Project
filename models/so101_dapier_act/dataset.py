"""SO-101 DAPIER-ACT 전용 HDF5 멀티모달 데이터셋 로더.

양팔 12 DOF + OS30A 3D PointCloud (512x3) + Left/Right Wrist RGB (240x320)
"""

import os
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DapierACTConfig
from .pointcloud import preprocess_os30a_depth_to_pointcloud


class DapierACTDataset(Dataset):
    """DAPIER-ACT 에피소드 데이터셋.

    각 샘플 반환 항목:
    - qpos: [12] 정규화된 현재 관절 각도
    - pointcloud: [512, 3] 미터 단위 작업영역 포인트
    - left_wrist: [3, 240, 320] RGB float32 in [0, 1]
    - right_wrist: [3, 240, 320] RGB float32 in [0, 1]
    - actions: [20, 12] 미래 20스텝 정규화된 액션 청크
    - is_pad: [20] 에피소드 종료 패딩 여부 (bool)
    """

    def __init__(
        self,
        dataset_dir: str,
        config: Optional[DapierACTConfig] = None,
        chunk_size: int = 20,
        stats: Optional[Dict[str, np.ndarray]] = None,
        cache_pointclouds: bool = True,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.cfg = config if config is not None else DapierACTConfig()
        self.chunk_size = chunk_size

        self.episode_files = sorted(self.dataset_dir.glob("episode_*.hdf5"))
        if not self.episode_files:
            raise FileNotFoundError(f"지정된 디렉토리에 episode_*.hdf5 파일이 없습니다: {dataset_dir}")

        # 에피소드별 프레임 인덱스 매핑 구성
        self.sample_indices = []  # List of (file_idx, timestep)
        self.episode_lengths = []

        for f_idx, ep_path in enumerate(self.episode_files):
            with h5py.File(ep_path, "r") as f:
                if "observations/qpos" in f:
                    ep_len = len(f["observations/qpos"])
                elif "action" in f:
                    ep_len = len(f["action"])
                else:
                    raise KeyError(f"{ep_path} 파일에 observations/qpos 또는 action 키가 없습니다.")

                self.episode_lengths.append(ep_len)
                for t in range(ep_len):
                    self.sample_indices.append((f_idx, t))

        # 통계 계산 또는 로드
        if stats is not None:
            self.stats = stats
        else:
            stats_path = self.dataset_dir / "dataset_stats.pkl"
            if stats_path.exists():
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
            else:
                self.stats = self.compute_statistics()
                with open(stats_path, "wb") as f:
                    pickle.dump(self.stats, f)

        # 포인트클라우드 인메모리 캐싱 (학습 에폭 속도 10배 향상)
        self.pointcloud_cache = {}
        if cache_pointclouds:
            self._build_pointcloud_cache()

    def compute_statistics(self) -> Dict[str, np.ndarray]:
        """데이터셋 전체에서 qpos/action의 mean, std 및 pointcloud min, max 계산"""
        all_qpos = []
        all_actions = []
        p_min = np.full(3, np.inf, dtype=np.float32)
        p_max = np.full(3, -np.inf, dtype=np.float32)

        t_base_camera = self.cfg.get_t_base_camera_matrix()

        print(f"\n📊 [{len(self.episode_files)}개 에피소드 데이터셋 통계 산출 중...]")
        for ep_path in self.episode_files:
            with h5py.File(ep_path, "r") as f:
                qpos = f["observations/qpos"][:]
                action = f["action"][:]
                all_qpos.append(qpos)
                all_actions.append(action)

                # 점군 범위 샘플링
                if "observations/pointcloud" in f:
                    pc = f["observations/pointcloud"][:]
                    p_min = np.minimum(p_min, pc.min(axis=(0, 1)))
                    p_max = np.maximum(p_max, pc.max(axis=(0, 1)))
                elif "observations/depth" in f:
                    depth_sample = f["observations/depth"][0]
                    pts = preprocess_os30a_depth_to_pointcloud(
                        depth_sample,
                        t_base_camera,
                        fx=self.cfg.fx,
                        fy=self.cfg.fy,
                        cx=self.cfg.cx,
                        cy=self.cfg.cy,
                        crop_x=self.cfg.crop_x,
                        crop_y=self.cfg.crop_y,
                        crop_z=self.cfg.crop_z,
                        num_points=self.cfg.num_point_samples,
                    )
                    p_min = np.minimum(p_min, pts.min(axis=0))
                    p_max = np.maximum(p_max, pts.max(axis=0))

        all_qpos = np.concatenate(all_qpos, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        qpos_mean = np.mean(all_qpos, axis=0)
        qpos_std = np.std(all_qpos, axis=0)
        qpos_std = np.clip(qpos_std, a_min=1e-2, a_max=None)

        action_mean = np.mean(all_actions, axis=0)
        action_std = np.std(all_actions, axis=0)
        action_std = np.clip(action_std, a_min=1e-2, a_max=None)

        if not np.isfinite(p_min).all() or not np.isfinite(p_max).all():
            p_min = np.array(self.cfg.default_point_min, dtype=np.float32)
            p_max = np.array(self.cfg.default_point_max, dtype=np.float32)

        stats = {
            "qpos_mean": qpos_mean,
            "qpos_std": qpos_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "point_min": p_min,
            "point_max": p_max,
        }
        print("✅ 데이터셋 정규화 통계 계산 완료.")
        return stats

    def _build_pointcloud_cache(self):
        """메모리에 512점 포인트클라우드 사전 캐싱"""
        t_base_camera = self.cfg.get_t_base_camera_matrix()
        print("⚡ [DP3 포인트클라우드 캐시 빌드 중...]")

        for f_idx, ep_path in enumerate(self.episode_files):
            with h5py.File(ep_path, "r") as f:
                ep_len = self.episode_lengths[f_idx]
                if "observations/pointcloud" in f:
                    for t in range(ep_len):
                        self.pointcloud_cache[(f_idx, t)] = f["observations/pointcloud"][t].astype(np.float32)
                elif "observations/depth" in f:
                    depth_seq = f["observations/depth"]
                    for t in range(ep_len):
                        d_img = depth_seq[t]
                        pts = preprocess_os30a_depth_to_pointcloud(
                            d_img,
                            t_base_camera,
                            fx=self.cfg.fx,
                            fy=self.cfg.fy,
                            cx=self.cfg.cx,
                            cy=self.cfg.cy,
                            crop_x=self.cfg.crop_x,
                            crop_y=self.cfg.crop_y,
                            crop_z=self.cfg.crop_z,
                            num_points=self.cfg.num_point_samples,
                        )
                        self.pointcloud_cache[(f_idx, t)] = pts

        print(f"✅ 총 {len(self.pointcloud_cache)}개 프레임 포인트클라우드 캐싱 완료.")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _load_image(self, ds, timestep: int) -> np.ndarray:
        """HDF5에서 이미지를 읽고 240x320 float32 RGB로 변환"""
        raw = ds[timestep]
        if isinstance(raw, (bytes, bytearray, np.void)) or (isinstance(raw, np.ndarray) and raw.ndim == 1):
            buf = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = raw
            if img.shape[-1] == 3 and ds.attrs.get("channel_order", "RGB") == "BGR":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] != self.cfg.wrist_height or img.shape[1] != self.cfg.wrist_width:
            img = cv2.resize(img, (self.cfg.wrist_width, self.cfg.wrist_height), interpolation=cv2.INTER_AREA)

        # [H, W, 3] -> [3, H, W], [0, 1] float32
        tensor = img.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_idx, timestep = self.sample_indices[idx]
        ep_path = self.episode_files[f_idx]
        ep_len = self.episode_lengths[f_idx]

        with h5py.File(ep_path, "r") as f:
            # 1. Qpos 정규화
            raw_qpos = f["observations/qpos"][timestep].astype(np.float32)
            norm_qpos = (raw_qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]

            # 2. 미래 20스텝 Action Chunk
            action_seq = f["action"]
            end_step = min(timestep + self.chunk_size, ep_len)
            raw_actions = action_seq[timestep:end_step].astype(np.float32)
            norm_actions = (raw_actions - self.stats["action_mean"]) / self.stats["action_std"]

            actual_len = norm_actions.shape[0]
            action_chunk = np.zeros((self.chunk_size, self.cfg.action_dim), dtype=np.float32)
            is_pad = np.ones(self.chunk_size, dtype=bool)

            action_chunk[:actual_len] = norm_actions
            is_pad[:actual_len] = False

            # 에피소드가 끝나면 마지막 액션으로 패딩
            if actual_len < self.chunk_size and actual_len > 0:
                action_chunk[actual_len:] = norm_actions[-1]

            # 3. 손목 카메라 이미지
            left_img = self._load_image(f["observations/images/left_wrist"], timestep)
            right_img = self._load_image(f["observations/images/right_wrist"], timestep)

            # 4. 포인트클라우드
            if (f_idx, timestep) in self.pointcloud_cache:
                pts = self.pointcloud_cache[(f_idx, timestep)]
            else:
                if "observations/pointcloud" in f:
                    pts = f["observations/pointcloud"][timestep].astype(np.float32)
                else:
                    d_img = f["observations/depth"][timestep]
                    pts = preprocess_os30a_depth_to_pointcloud(
                        d_img,
                        self.cfg.get_t_base_camera_matrix(),
                        fx=self.cfg.fx,
                        fy=self.cfg.fy,
                        cx=self.cfg.cx,
                        cy=self.cfg.cy,
                        crop_x=self.cfg.crop_x,
                        crop_y=self.cfg.crop_y,
                        crop_z=self.cfg.crop_z,
                        num_points=self.cfg.num_point_samples,
                    )

        return {
            "qpos": torch.from_numpy(norm_qpos).float(),
            "pointcloud": torch.from_numpy(pts).float(),
            "left_wrist": torch.from_numpy(left_img).float(),
            "right_wrist": torch.from_numpy(right_img).float(),
            "actions": torch.from_numpy(action_chunk).float(),
            "is_pad": torch.from_numpy(is_pad),
        }
