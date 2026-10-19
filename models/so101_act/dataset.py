"""SO-101 픽앤플레이스 HDF5 데이터 로더 및 정규화 모듈."""

import glob
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class SO101EpisodicDataset(Dataset):
    """SO-101 6-DOF 데모 에피소드 데이터셋 (멀티카메라 + 액션 청킹)."""

    def __init__(
        self,
        episode_paths: list,
        camera_names: list,
        norm_stats: dict,
        chunk_size: int = 50,
    ):
        super().__init__()
        self.episode_paths = episode_paths
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size

        # ImageNet 정규화
        self.normalize_img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.episode_paths)

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

            # 카메라 이미지 로드 (RGB, uint8)
            cam_images = []
            for cam_name in self.camera_names:
                img = root[f"observations/images/{cam_name}"][start_ts]  # (480, 640, 3)
                # (H, W, C) -> (C, H, W)
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_t = self.normalize_img(img_t)
                cam_images.append(img_t)

            # 멀티카메라 텐서 결합 (num_cam, 3, 480, 640)
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
                # 마지막 액션 복사 유지
                padded_action[actual_len:] = action_slice[-1]
                is_pad[actual_len:] = 1.0  # 마스킹 구간

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

    all_qpos = np.concatenate(all_qpos, axis=0)  # (Total_T, 6)
    all_actions = np.concatenate(all_actions, axis=0)  # (Total_T, 6)

    qpos_mean = np.mean(all_qpos, axis=0)
    qpos_std = np.std(all_qpos, axis=0)
    qpos_std = np.clip(qpos_std, 1e-2, np.inf)

    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    action_std = np.clip(action_std, 1e-2, np.inf)

    stats = {
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    return stats


def parse_episode_filter(filter_str: str) -> set:
    """문자열('56-76' 또는 '1,2,5-10')을 에피소드 정수 번호 집합으로 파싱."""
    if not filter_str:
        return None
    selected = set()
    for part in filter_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s, e = part.split("-")
            selected.update(range(int(s), int(e) + 1))
        else:
            selected.add(int(part))
    return selected


def load_so101_dataloaders(
    data_dir: str,
    camera_names: list,
    batch_size: int = 8,
    chunk_size: int = 50,
    train_ratio: float = 0.8,
    seed: int = 0,
    episodes: str = None,
):
    """HDF5 에피소드 파일들을 탐색하여 Train/Val 데이터로더 및 통계 반환"""
    all_paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    selected_indices = parse_episode_filter(episodes)

    if selected_indices is not None:
        episode_paths = []
        for p in all_paths:
            base = os.path.splitext(os.path.basename(p))[0]
            try:
                ep_num = int(base.split("_")[-1])
                if ep_num in selected_indices:
                    episode_paths.append(p)
            except ValueError:
                pass
    else:
        episode_paths = all_paths

    num_episodes = len(episode_paths)
    if num_episodes == 0:
        raise FileNotFoundError(f"'{data_dir}'에서 조건(episodes={episodes})에 맞는 HDF5 에피소드를 찾을 수 없습니다.")

    # 재현성을 위한 셔플
    np.random.seed(seed)
    shuffled_paths = np.random.permutation(episode_paths)

    num_train = max(1, int(num_episodes * train_ratio))
    train_paths = shuffled_paths[:num_train].tolist()
    val_paths = shuffled_paths[num_train:].tolist()

    # 정규화 통계는 전체 데이터셋으로부터 계산
    norm_stats = get_dataset_norm_stats(episode_paths)

    train_dataset = SO101EpisodicDataset(train_paths, camera_names, norm_stats, chunk_size=chunk_size)
    val_dataset = SO101EpisodicDataset(val_paths, camera_names, norm_stats, chunk_size=chunk_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=1,
    )

    print(f"\n📊 [데이터셋 로드 완료]")
    print(f" - 전체 에피소드: {num_episodes}개")
    print(f" - 학습용(Train): {len(train_paths)}개 / 검증용(Val): {len(val_paths)}개")
    print(f" - 활성 카메라: {camera_names} | 청크 크기: {chunk_size}스텝")

    return train_loader, val_loader, norm_stats
