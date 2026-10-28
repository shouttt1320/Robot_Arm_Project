"""OS30A 뎁스 카메라 인터페이스 (YDLIDAR OS30A / HP-ASC-H201).

기능:
1. 실시간 OS30A 카메라 스트림 연결 (또는 테스트용 샘플 리플레이 모드)
2. 640x460 uint16 (mm 단위) 뎁스 프레임 취득
3. 카메라 캘리브레이션 K 및 Extrinsic T를 통한 512점 DP3 3D 포인트클라우드 실시간 추출
"""

import os
from pathlib import Path
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from models.so101_dapier_act.config import DapierACTConfig
from models.so101_dapier_act.pointcloud import (
    preprocess_os30a_depth_to_pointcloud,
    depth_to_xyz,
    transform_points,
    crop_xyz,
    farthest_point_sample_numpy,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "upstream_models/DAPIER-ACT/adapters/sweepick_261020_os30a_data/sample_20261020_094913"


class OS30ACamera:
    """YDLIDAR OS30A 뎁스 카메라 제어기"""

    def __init__(
        self,
        config: Optional[DapierACTConfig] = None,
        use_sample_fallback: bool = True,
        device_index: Optional[int] = None,
    ):
        self.cfg = config if config is not None else DapierACTConfig()
        self.use_sample_fallback = use_sample_fallback
        self.device_index = device_index
        self.t_base_camera = self.cfg.get_t_base_camera_matrix()

        self.cap = None
        self.is_live = False
        self._sample_depth = None

        self._init_camera()

    def _init_camera(self):
        """카메라 연결 시도"""
        # 1. 사용자가 지정한 디바이스 인덱스 또는 기본 V4L2 탐색
        if self.device_index is not None:
            self.cap = cv2.VideoCapture(self.device_index)
            if self.cap.isOpened():
                self.is_live = True
                print(f"✅ [OS30A] /dev/video{self.device_index} 연결 성공!")
                return

        # 2. 실물 카메라가 아직 없거나 연결 실패 시 샘플 데이터 폴백 로드
        if self.use_sample_fallback and SAMPLE_DIR.exists():
            sample_png = SAMPLE_DIR / "depth_000006.png"
            sample_npy = SAMPLE_DIR / "depth_mm.npy"
            if sample_png.exists():
                self._sample_depth = cv2.imread(str(sample_png), cv2.IMREAD_UNCHANGED)
                print(f"ℹ️ [OS30A] 실물 카메라 미연결 상태: 샘플 뎁스 프레임 로드 완료 ({sample_png.name})")
            elif sample_npy.exists():
                self._sample_depth = np.load(str(sample_npy)).astype(np.uint16)
                print(f"ℹ️ [OS30A] 샘플 뎁스 프레임 로드 완료 ({sample_npy.name})")

    def read_depth(self) -> Tuple[bool, np.ndarray]:
        """640x460 uint16 (mm 단위) 뎁스 프레임 반환"""
        if self.is_live and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # 단일 채널 16비트 변환
                if frame.ndim == 3:
                    frame = frame[:, :, 0]
                if frame.shape != (self.cfg.depth_height, self.cfg.depth_width):
                    frame = cv2.resize(
                        frame, (self.cfg.depth_width, self.cfg.depth_height), interpolation=cv2.INTER_NEAREST
                    )
                return True, frame.astype(np.uint16)

        if self._sample_depth is not None:
            return True, self._sample_depth.copy()

        return False, np.zeros((self.cfg.depth_height, self.cfg.depth_width), dtype=np.uint16)

    def read_pointcloud(self) -> Tuple[bool, np.ndarray]:
        """작업영역 자르기 및 FPS 512점이 완료된 [512, 3] 포인트클라우드 반환"""
        ret, depth = self.read_depth()
        if not ret:
            return False, np.zeros((self.cfg.num_point_samples, 3), dtype=np.float32)

        pts = preprocess_os30a_depth_to_pointcloud(
            depth,
            self.t_base_camera,
            fx=self.cfg.fx,
            fy=self.cfg.fy,
            cx=self.cfg.cx,
            cy=self.cfg.cy,
            depth_scale_to_m=self.cfg.depth_scale_to_m,
            crop_x=self.cfg.crop_x,
            crop_y=self.cfg.crop_y,
            crop_z=self.cfg.crop_z,
            num_points=self.cfg.num_point_samples,
            candidate_points=self.cfg.candidate_points,
        )
        return True, pts

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_live = False
