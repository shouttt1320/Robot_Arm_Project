"""SO-101 DAPIER-ACT 설정 및 기본 하이퍼파라미터 정의.

양팔(Dual SO-101, 12 DOF) + OS30A Head Depth (DP3 PointNet) + Left/Right Wrist RGB (ResNet-18)
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class DapierACTConfig:
    # ----------------------------------------------------
    # 로봇 관절 및 액션 차원 (양팔 12 DOF)
    # Left 6 DOF (5 관절 + 1 그리퍼) + Right 6 DOF (5 관절 + 1 그리퍼)
    # ----------------------------------------------------
    state_dim: int = 12
    action_dim: int = 12
    chunk_size: int = 20  # 액션 청크 크기 (미래 20스텝 동시 예측)
    hidden_dim: int = 256  # 트랜스포머 은닉 차원
    latent_dim: int = 32  # CVAE 잠재 공간 차원

    # ----------------------------------------------------
    # 트랜스포머 아키텍처
    # ----------------------------------------------------
    nheads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    cvae_encoder_layers: int = 2
    ff_dim: int = 1024
    dropout: float = 0.0
    kl_weight: float = 10.0  # CVAE KL divergence 가중치

    # ----------------------------------------------------
    # OS30A 뎁스 카메라 및 DP3 포인트클라우드 규격
    # ----------------------------------------------------
    depth_width: int = 640
    depth_height: int = 460
    depth_scale_to_m: float = 0.001 / 1.016  # mm -> m 변환 및 실측 1.6% 거리 오차 보정

    # OS30A 공장 정류 내부 파라미터 K (640x460 정류 depth 기준)
    fx: float = 409.3
    fy: float = 409.3
    cx: float = 313.14875
    cy: float = 231.20775

    # 작업영역(Task Frame) 바운딩 박스 자르기 범위 (미터 단위)
    crop_x: Tuple[float, float] = (0.15, 0.32)
    crop_y: Tuple[float, float] = (-0.26, 0.26)
    crop_z: Tuple[float, float] = (0.005, 0.40)

    # DP3 포인트클라우드 샘플링
    candidate_points: int = 8192
    num_point_samples: int = 512  # FPS로 추출할 대표 점 개수
    point_tokens: int = 1  # PointNet 글로벌 풀링 출력 토큰 수

    # 기본 정규화 범위 (학습 데이터셋 통계로 업데이트 가능)
    default_point_min: Tuple[float, float, float] = (0.15, -0.26, 0.005)
    default_point_max: Tuple[float, float, float] = (0.32, 0.26, 0.40)

    # OS30A Extrinsics (T_base_footprint_from_os30a_optical)
    # p_base = R @ p_optical + t
    t_base_camera: List[float] = field(
        default_factory=lambda: [-0.0330105670802942, 0.022067149295795792, 0.5129377312368042]
    )
    r_base_camera: List[List[float]] = field(
        default_factory=lambda: [
            [0.06336692873381891, -0.9349076663440383, 0.3491880406800661],
            [-0.9978120697194266, -0.06596322063031296, 0.0044639720323631305],
            [0.01886016609344396, -0.34870690978989316, -0.9370420402520407],
        ]
    )

    # ----------------------------------------------------
    # 손목 카메라 (Left Wrist, Right Wrist) 규격
    # ----------------------------------------------------
    wrist_width: int = 320
    wrist_height: int = 240
    wrist_tokens_per_camera: int = 80  # (240/32) * (320/32) = 8 * 10 = 80
    pretrained_wrist_backbone: bool = True
    shared_wrist_backbone: bool = False  # False: 좌/우 독립 ResNet-18 사용

    # ----------------------------------------------------
    # 총 트랜스포머 Context 토큰 수 (163개)
    # Latent (1) + Qpos (1) + Head PointCloud (1) + Left Wrist (80) + Right Wrist (80)
    # ----------------------------------------------------
    @property
    def total_context_tokens(self) -> int:
        return 1 + 1 + self.point_tokens + (self.wrist_tokens_per_camera * 2)

    # ----------------------------------------------------
    # 학습 설정
    # ----------------------------------------------------
    batch_size: int = 16
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 2000
    seed: int = 42

    def get_t_base_camera_matrix(self) -> np.ndarray:
        """4x4 동차 변환 행렬 생성"""
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.array(self.r_base_camera, dtype=np.float32)
        mat[:3, 3] = np.array(self.t_base_camera, dtype=np.float32)
        return mat
