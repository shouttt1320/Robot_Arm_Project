"""OS30A 뎁스 영상 처리 및 DP3 기반 3D 포인트클라우드 전처리 모듈.

파이프라인:
Depth (uint16 mm) -> Camera Frame XYZ (m) -> Base/Task Frame 변환 -> 작업영역 Crop -> 후보군 축소 -> FPS 512점 -> 정규화
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch


def depth_to_xyz(
    depth: np.ndarray,
    fx: float = 409.3,
    fy: float = 409.3,
    cx: float = 313.14875,
    cy: float = 231.20775,
    depth_scale_to_m: float = 0.001 / 1.016,
) -> np.ndarray:
    """단일 2D 뎁스 맵(HxW)을 카메라 좌표계 기준 Nx3 3D 포인트(미터 단위)로 역투영"""
    if depth.ndim != 2:
        raise ValueError(f"뎁스 맵은 2차원(H, W)이어야 합니다. 현재 형태: {depth.shape}")

    # 유효하지 않은 특수 코드(16384 등) 필터링
    valid_mask = (depth > 0) & (depth < 16000)

    z = depth.astype(np.float32) * np.float32(depth_scale_to_m)
    h, w = z.shape

    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    x = (u - np.float32(cx)) * z / np.float32(fx)
    y = (v - np.float32(cy)) * z / np.float32(fy)

    pts = np.stack([x, y, z], axis=-1)
    pts_valid = pts[valid_mask]
    return pts_valid.astype(np.float32)


def transform_points(t_mat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """4x4 동차 변환 행렬을 적용하여 좌표계 변환: p_out = R @ p_in + t"""
    if points.shape[0] == 0:
        return points
    r = t_mat[:3, :3]
    t = t_mat[:3, 3]
    return (points @ r.T) + t


def crop_xyz(
    points: np.ndarray,
    crop_x: Tuple[float, float] = (0.15, 0.32),
    crop_y: Tuple[float, float] = (-0.26, 0.26),
    crop_z: Tuple[float, float] = (0.005, 0.40),
) -> np.ndarray:
    """작업영역(Task Frame) 바운딩 박스 내부의 점만 필터링"""
    if len(points) == 0:
        return points

    mask = (
        (points[:, 0] >= crop_x[0]) & (points[:, 0] <= crop_x[1]) &
        (points[:, 1] >= crop_y[0]) & (points[:, 1] <= crop_y[1]) &
        (points[:, 2] >= crop_z[0]) & (points[:, 2] <= crop_z[1])
    )
    return points[mask]


def linspace_candidates(xyz: torch.Tensor, max_candidates: int = 8192) -> torch.Tensor:
    """포인트가 너무 많을 경우 FPS 연산량 절감을 위해 균일 인덱스로 후보군 압축"""
    n = xyz.shape[0]
    if n <= max_candidates:
        return xyz
    indices = torch.linspace(0, n - 1, max_candidates, device=xyz.device).round().long()
    return xyz.index_select(0, indices)


def farthest_point_sample(xyz: torch.Tensor, num_samples: int = 512) -> torch.Tensor:
    """미터 공간에서 가장 먼 점을 차례로 선택하는 Farthest Point Sampling (PyTorch 텐서)"""
    n = xyz.shape[0]
    k = min(num_samples, n)

    selected = torch.empty(k, dtype=torch.long, device=xyz.device)
    min_dist = torch.full((n,), float("inf"), device=xyz.device)

    # 중심점에서 가장 먼 점부터 시작
    center = xyz.mean(dim=0, keepdim=True)
    farthest = torch.argmax(torch.sum((xyz - center) ** 2, dim=1))

    for i in range(k):
        selected[i] = farthest
        point = xyz[farthest].unsqueeze(0)
        dist = torch.sum((xyz - point) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist)

    sampled = xyz.index_select(0, selected)

    # 점이 부족한 경우 복제 패딩
    if k < num_samples:
        repeat_count = (num_samples + k - 1) // k
        sampled = sampled.repeat(repeat_count, 1)[:num_samples]

    return sampled


def farthest_point_sample_numpy(points: np.ndarray, num_samples: int = 512) -> np.ndarray:
    """NumPy 전용 FPS 구현"""
    if len(points) == 0:
        return np.zeros((num_samples, 3), dtype=np.float32)
    t_pts = torch.from_numpy(points).float()
    sampled = farthest_point_sample(t_pts, num_samples)
    return sampled.numpy()


def limits_normalize(
    xyz: Union[np.ndarray, torch.Tensor],
    point_min: Union[np.ndarray, torch.Tensor],
    point_max: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-4,
) -> Union[np.ndarray, torch.Tensor]:
    """[point_min, point_max] 범위를 [-1, 1] 구간으로 선형 정규화"""
    if isinstance(xyz, torch.Tensor):
        p_min = torch.as_tensor(point_min, dtype=xyz.dtype, device=xyz.device)
        p_max = torch.as_tensor(point_max, dtype=xyz.dtype, device=xyz.device)
        p_range = (p_max - p_min).clamp_min(eps)
        return 2.0 * (xyz - p_min) / p_range - 1.0
    else:
        p_min = np.asarray(point_min, dtype=np.float32)
        p_max = np.asarray(point_max, dtype=np.float32)
        p_range = np.maximum(p_max - p_min, eps)
        return 2.0 * (xyz - p_min) / p_range - 1.0


def preprocess_os30a_depth_to_pointcloud(
    depth: np.ndarray,
    t_base_camera: np.ndarray,
    fx: float = 409.3,
    fy: float = 409.3,
    cx: float = 313.14875,
    cy: float = 231.20775,
    depth_scale_to_m: float = 0.001 / 1.016,
    crop_x: Tuple[float, float] = (0.15, 0.32),
    crop_y: Tuple[float, float] = (-0.26, 0.26),
    crop_z: Tuple[float, float] = (0.005, 0.40),
    num_points: int = 512,
    candidate_points: int = 8192,
) -> np.ndarray:
    """원시 OS30A 뎁스 맵 한 장을 Task Frame 512개 3D 포인트([512, 3])로 완결 변환"""
    # 1. 뎁스 -> 카메라 좌표계 XYZ
    pts_cam = depth_to_xyz(depth, fx, fy, cx, cy, depth_scale_to_m)
    if len(pts_cam) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    # 2. 카메라 좌표계 -> 로봇 Base/Task 좌표계
    pts_task = transform_points(t_base_camera, pts_cam)

    # 3. 작업영역 Crop
    pts_crop = crop_xyz(pts_task, crop_x, crop_y, crop_z)
    if len(pts_crop) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    # 4. 최대 후보군 축소
    if len(pts_crop) > candidate_points:
        indices = np.linspace(0, len(pts_crop) - 1, candidate_points).round().astype(int)
        pts_crop = pts_crop[indices]

    # 5. FPS 512점 샘플링
    sampled = farthest_point_sample_numpy(pts_crop, num_points)
    return sampled.astype(np.float32)
