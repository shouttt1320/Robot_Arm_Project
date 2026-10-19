"""지연(Latency) 없는 멀티스레드 멀티카메라 캡처 인터페이스.

일반 OpenCV cap.read()의 내부 버퍼 지연(Lag)을 방지하기 위해,
각 카메라마다 백그라운드 스레드를 실행하여 항상 최신 프레임을 실시간 유지합니다.
"""

import threading
import time
from typing import Dict, List, Optional
import cv2
import numpy as np


class ThreadedCamera:
    """단일 카메라 전용 고속 백그라운드 캡처 스레드"""

    def __init__(self, device_id: int, width: int = 640, height: int = 480, fps: int = 30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 /dev/video{self.device_id} 를 열 수 없습니다.")

        # 해상도 및 FPS 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 내부 버퍼 크기 최소화 (지연 제거)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

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
    """여러 대의 카메라(Top, Wrist 등)를 통합 관리하는 인터페이스"""

    def __init__(self, camera_dict: Dict[str, int], width: int = 640, height: int = 480, fps: int = 30):
        """
        camera_dict: {'top': 4, 'wrist': 6} 형식의 카메라 이름-디바이스ID 딕셔너리
        """
        self.camera_dict = camera_dict
        self.width = width
        self.height = height
        self.fps = fps
        self.cameras: Dict[str, ThreadedCamera] = {}

    def start(self):
        print("\n[카메라 연결 시작]")
        for name, dev_id in self.camera_dict.items():
            print(f" - [{name}] 카메라 초기화 중... (/dev/video{dev_id})")
            cam = ThreadedCamera(dev_id, width=self.width, height=self.height, fps=self.fps)
            cam.start()
            self.cameras[name] = cam
            print(f"   ✅ [{name}] 연결 성공!")

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
