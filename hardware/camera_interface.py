"""지연(Latency) 없는 멀티스레드 멀티카메라 캡처 인터페이스.

일반 OpenCV cap.read()의 내부 버퍼 지연(Lag)을 방지하기 위해,
각 카메라마다 백그라운드 스레드를 실행하여 항상 최신 프레임을 실시간 유지합니다.
"""

import glob
import os
import threading
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def auto_detect_camera_indices(default_top: int = 4, default_wrist: int = 7) -> Tuple[int, int]:
    """시스템에 연결된 Top/Wrist 카메라의 video 디바이스 번호를 이름 기반 자동 탐색"""
    top_id, wrist_id = None, None
    for dev_path in sorted(glob.glob('/sys/class/video4linux/video*'), key=lambda p: int(os.path.basename(p).replace('video', ''))):
        try:
            idx = int(os.path.basename(dev_path).replace('video', ''))
            with open(os.path.join(dev_path, 'name'), 'r', encoding='utf-8', errors='ignore') as f:
                name = f.read().strip().lower()
            if 'innomaker' in name:
                if wrist_id is None:
                    wrist_id = idx
            elif ('pc cam' in name or 'usb 2.0 camera' in name) and 'asus' not in name:
                if top_id is None:
                    top_id = idx
        except Exception:
            pass

    final_top = top_id if top_id is not None else default_top
    final_wrist = wrist_id if wrist_id is not None else default_wrist
    return final_top, final_wrist


class ThreadedCamera:
    """단일 카메라 전용 고속 백그라운드 캡처 스레드 (초점 고정 기능 포함)"""

    def __init__(
        self,
        device_id: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        disable_autofocus: bool = True,
        focus_value: Optional[int] = 25,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.disable_autofocus = disable_autofocus
        self.focus_value = focus_value

        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 /dev/video{self.device_id} 를 열 수 없습니다.")

        # 해상도 및 FPS 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 내부 버퍼 크기 최소화 (지연 제거)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 오토포커스 비활성화 및 초점 고정 (학습 데이터 이미지 일관성 유지)
        if self.disable_autofocus:
            self.set_autofocus(False)
            if self.focus_value is not None:
                self.set_focus(self.focus_value)

        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def set_autofocus(self, enable: bool) -> bool:
        """오토포커스 On/Off (False: 오토포커스 해제 및 수동 고정)"""
        val = 1 if enable else 0
        return self.cap.set(cv2.CAP_PROP_AUTOFOCUS, val)

    def set_focus(self, focus_val: int) -> bool:
        """수동 초점 값 설정 (0~255) 및 오토포커스 해제"""
        self.set_autofocus(False)
        return self.cap.set(cv2.CAP_PROP_FOCUS, float(focus_val))

    def get_focus(self) -> float:
        """현재 초점 값 조회"""
        return self.cap.get(cv2.CAP_PROP_FOCUS)

    def get_autofocus(self) -> bool:
        """현재 오토포커스 활성화 여부 조회"""
        return bool(self.cap.get(cv2.CAP_PROP_AUTOFOCUS))

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        # 첫 번째 프레임이 읽힐 때까지 최대 2초 대기
        start_t = time.time()
        while self.frame is None and time.time() - start_t < 2.0:
            time.sleep(0.02)
        if self.frame is None:
            raise RuntimeError(f"카메라 /dev/video{self.device_id} 에서 첫 프레임을 읽지 못했습니다.")

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


class MultiCameraCapture:
    """여러 대의 카메라(Top, Wrist, 양안 등)를 통합 관리하는 인터페이스 (초점 고정 지원)"""

    def __init__(
        self,
        camera_dict: Dict[str, int],
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        disable_autofocus: bool = True,
        focus_dict: Optional[Dict[str, int]] = None,
    ):
        """
        camera_dict: {'cam_left': 4, 'cam_right': 6} 형식의 카메라 이름-디바이스ID 딕셔너리
        disable_autofocus: 오토포커스 비활성화 및 초점 고정 여부
        focus_dict: {'cam_left': 25, 'cam_right': 25} 형식의 카메라별 고정 초점 값
        """
        self.camera_dict = camera_dict
        self.width = width
        self.height = height
        self.fps = fps
        self.disable_autofocus = disable_autofocus
        self.focus_dict = focus_dict or {}
        self.cameras: Dict[str, ThreadedCamera] = {}

    def start(self):
        print("\n[카메라 연결 시작]")
        for name, dev_id in list(self.camera_dict.items()):
            # 디바이스 경로가 없으면 자동 탐색 폴백
            dev_path = f"/dev/video{dev_id}"
            if not os.path.exists(dev_path):
                auto_top, auto_wrist = auto_detect_camera_indices()
                fallback_id = auto_wrist if name == "wrist" else auto_top
                print(f"⚠️ [{name}] {dev_path} 없음 -> 자동 탐색된 /dev/video{fallback_id} 로 대체 연결합니다.")
                dev_id = fallback_id
                self.camera_dict[name] = dev_id

            target_focus = self.focus_dict.get(name, 25)
            print(f" - [{name}] 카메라 초기화 중... (/dev/video{dev_id}, Focus 고정={self.disable_autofocus}, Focus값={target_focus})")
            cam = ThreadedCamera(
                dev_id,
                width=self.width,
                height=self.height,
                fps=self.fps,
                disable_autofocus=self.disable_autofocus,
                focus_value=target_focus,
            )
            cam.start()
            self.cameras[name] = cam
            cur_focus = cam.get_focus()
            cur_af = "AUTO" if cam.get_autofocus() else "LOCKED"
            print(f"   ✅ [{name}] 연결 성공! (초점 상태: {cur_af}, 현재 Focus: {cur_focus})")

    def read_all(self) -> Dict[str, Optional[np.ndarray]]:
        """모든 카메라의 최신 프레임을 딕셔너리로 반환"""
        frames = {}
        for name, cam in self.cameras.items():
            frames[name] = cam.read()
        return frames

    def stop(self):
        print("\n모든 카메라 연결을 해제합니다...")
        for name, cam in self.cameras.items():
            cam.stop()
        self.cameras.clear()
        print("카메라 정리 완료.")
