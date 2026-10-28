"""양안(Dual Eyes) 카메라 HDF5 데이터 로더 및 전처리 모듈."""

import glob
import os
from typing import Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class DualEyesEpisodicDataset(Dataset):
    """양안(좌/우) 카메라 및 6-DOF 로봇 데모 에피소드 데이터셋."""

    def __init__(
        self,
        episode_paths: list,
        camera_names: list,
        norm_stats: dict,
        chunk_size: int = 50,
        image_size: Tuple[int, int] = (224, 294),
    ):
        super().__init__()
        self.episode_paths = episode_paths
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.image_size = image_size

        # DINOv2 표준 전처리 파이프라인 (Bicubic Resize + ImageNet Normalization)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.episode_paths)

    def _get_camera_image(self, root: h5py.File, cam_name: str, step: int) -> np.ndarray:
        """HDF5 내부 경로 호환 처리 (cam_left, left, cam_0 등 유연 대응)."""
        possible_keys = [
            f"observations/images/{cam_name}",
            f"observations/images/{cam_name.replace('cam_', '')}",
            f"observations/images/cam_{cam_name}",
        ]
        for key in possible_keys:
            if key in root:
                return root[key][step]
        # 키를 못 찾은 경우 images 하위의 사용 가능한 키 확인
        if "observations/images" in root:
            available = list(root["observations/images"].keys())
            raise KeyError(f"카메라 '{cam_name}'를 찾을 수 없습니다. 사용 가능한 키: {available}")
        raise KeyError(f"HDF5에 'observations/images' 경로가 없습니다.")

    def __getitem__(self, index):
        hdf5_path = self.episode_paths[index]

        with h5py.File(hdf5_path, "r") as root:
            actions = root["action"][()]  # (T, 6)
            qpos_all = root["observations/qpos"][()]  # (T, 6)
            episode_len = actions.shape[0]

            # 1. 무작위 시작 시점 샘플링
            start_ts = np.random.choice(episode_len)

            # 2. start_ts 시점의 관측값 (Observation)
            qpos = qpos_all[start_ts]

            # 양안 카메라 이미지 로드 (RGB, uint8)
            cam_images = []
            for cam_name in self.camera_names:
                img = self._get_camera_image(root, cam_name, start_ts)  # (H, W, 3)
                # (H, W, C) -> (C, H, W) float [0, 1]
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                # Bicubic Resize (14배수) + Normalize
                img_t = self.transform(img_t)
                cam_images.append(img_t)

            # 양안 텐서 결합 (num_cam, 3, H, W)
            image_data = torch.stack(cam_images, dim=0)

            # 3. start_ts 부터 미래 chunk_size(50)스텝 액션 슬라이싱
            end_ts = start_ts + self.chunk_size
            action_slice = actions[start_ts:end_ts]
            actual_len = action_slice.shape[0]

            # 미래 스텝이 부족한 경우 마지막 액션으로 패딩하고 is_pad 마스킹
            padded_action = np.zeros((self.chunk_size, 6), dtype=np.float32)
            padded_action[:actual_len] = action_slice

            is_pad = np.zeros(self.chunk_size, dtype=np.float32)
            if actual_len < self.chunk_size:
                padded_action[actual_len:] = action_slice[-1]
                is_pad[actual_len:] = 1.0

        # 4. 정규화 적용 (qpos 및 action)
        qpos_norm = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_norm = (padded_action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        qpos_data = torch.from_numpy(qpos_norm).float()
        action_data = torch.from_numpy(action_norm).float()
        is_pad_data = torch.from_numpy(is_pad).bool()

        return image_data, qpos_data, action_data, is_pad_data


def get_dataset_norm_stats(episode_paths: list) -> dict:
    """모든 에피소드 데이터로부터 관절 각도 및 액션의 평균/표준편차 통계 산출."""
    all_qpos = []
    all_actions = []

    for path in episode_paths:
        with h5py.File(path, "r") as root:
            q = root["observations/qpos"][()]  # (T, 6)
            a = root["action"][()]  # (T, 6)
            all_qpos.append(q)
            all_actions.append(a)

    all_qpos = np.concatenate(all_qpos, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    stats = {
        "qpos_mean": np.mean(all_qpos, axis=0).astype(np.float32),
        "qpos_std": np.clip(np.std(all_qpos, axis=0), 1e-4, np.inf).astype(np.float32),
        "action_mean": np.mean(all_actions, axis=0).astype(np.float32),
        "action_std": np.clip(np.std(all_actions, axis=0), 1e-4, np.inf).astype(np.float32),
    }
    return stats


def load_dual_eyes_dataloaders(
    data_dir: str,
    camera_names: list,
    batch_size: int = 8,
    chunk_size: int = 50,
    image_size: Tuple[int, int] = (224, 294),
    val_ratio: float = 0.0,
    num_workers: int = 4,
):
    """양안 데이터셋 로더 및 통계 반환 함수."""
    episode_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not episode_files:
        raise FileNotFoundError(f"'{data_dir}' 경로에서 HDF5 파일을 찾을 수 없습니다.")

    n_episodes = len(episode_files)
    if val_ratio <= 0.0:
        # 모든 에피소드를 100% 학습에 사용
        train_episodes = episode_files
        val_episodes = episode_files
        print(f"📊 [Dataset] 모든 에피소드({n_episodes}개, 100%)를 학습(Train)에 사용합니다.")
    else:
        # Train / Val 분할
        n_val = max(1, int(n_episodes * val_ratio))
        val_episodes = episode_files[:n_val]
        train_episodes = episode_files[n_val:]
        print(f"[Dataset] 총 {n_episodes}개 에피소드 중 {len(train_episodes)}개 Train, {len(val_episodes)}개 Val 분할")

    # 통계치 산출
    norm_stats = get_dataset_norm_stats(train_episodes)

    train_dataset = DualEyesEpisodicDataset(
        train_episodes, camera_names, norm_stats, chunk_size=chunk_size, image_size=image_size
    )
    val_dataset = DualEyesEpisodicDataset(
        val_episodes, camera_names, norm_stats, chunk_size=chunk_size, image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, norm_stats
