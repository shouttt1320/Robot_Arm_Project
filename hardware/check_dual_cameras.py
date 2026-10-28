"""양안(Dual Eyes) 스테레오 카메라 실시간 시점, 좌/우 식별 및 초점 고정(Auto Focus Lock) 점검 도구.

사용 목적:
1. 두 대의 카메라가 물리적으로 왼쪽/오른쪽 시점에 올바르게 배치되었는지 나란히(Side-by-Side) 비교 확인.
2. 수평 정렬 가이드라인(Grid)을 통해 양안 카메라의 롤(Roll)/피치(Pitch) 수평선 일치 여부 점검.
3. [초점 고정(Focus Lock)]: 학습 데이터 수집 시 렌즈의 숨쉬기/초점 변동 방지를 위해 오토포커스 비활성화 및 초점값 고정.
4. 'f' 키로 원샷 오토포커스(Auto Focus 후 고정), '+/-' 키로 수동 초점 미세조정.
5. 's' 키로 즉시 좌/우 카메라 스왑(Swap).
6. 'c' 키로 캘리브레이션용 스냅샷 저장.
"""

import argparse
import glob
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def list_available_cameras() -> List[Tuple[int, str]]:
    """시스템에 연결된 모든 V4L2 카메라 인덱스와 디바이스 이름을 검색합니다."""
    cams = []
    video_paths = sorted(
        glob.glob("/sys/class/video4linux/video*"),
        key=lambda p: int(os.path.basename(p).replace("video", "")),
    )
    for dev_path in video_paths:
        try:
            idx = int(os.path.basename(dev_path).replace("video", ""))
            name_file = os.path.join(dev_path, "name")
            name = "Unknown Camera"
            if os.path.exists(name_file):
                with open(name_file, "r", encoding="utf-8", errors="ignore") as f:
                    name = f.read().strip()
            cams.append((idx, name))
        except Exception:
            pass
    return cams


class CameraStream:
    """단일 카메라 비동기 캡처 및 초점(Focus) 고정 제어 클래스."""

    def __init__(
        self,
        device_id: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        disable_autofocus: bool = True,
        initial_focus: Optional[int] = 25,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.disable_autofocus = disable_autofocus
        self.focus_value = initial_focus
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.is_opened = False
        self._init_camera()

    def _init_camera(self):
        # V4L2 백엔드로 열기
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_id)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 지연 최소화

            # 오토포커스 비활성화 및 초점 고정
            if self.disable_autofocus:
                self.set_autofocus(False)
                if self.focus_value is not None:
                    self.set_focus(self.focus_value)

            self.is_opened = True
        else:
            self.is_opened = False

    def set_autofocus(self, enable: bool) -> bool:
        """오토포커스 켜기/끄기 (False: 오토포커스 해제 및 수동 고정)"""
        if not self.is_opened or self.cap is None:
            return False
        val = 1 if enable else 0
        return self.cap.set(cv2.CAP_PROP_AUTOFOCUS, val)

    def set_focus(self, focus_val: int) -> bool:
        """수동 초점 값 설정 (0~255) 및 오토포커스 해제"""
        if not self.is_opened or self.cap is None:
            return False
        focus_val = int(np.clip(focus_val, 0, 255))
        self.set_autofocus(False)
        res = self.cap.set(cv2.CAP_PROP_FOCUS, float(focus_val))
        if res:
            self.focus_value = focus_val
        return res

    def get_focus(self) -> float:
        """현재 초점 값 조회"""
        if not self.is_opened or self.cap is None:
            return -1.0
        return self.cap.get(cv2.CAP_PROP_FOCUS)

    def get_autofocus(self) -> bool:
        """현재 오토포커스 활성화 여부 조회"""
        if not self.is_opened or self.cap is None:
            return False
        return bool(self.cap.get(cv2.CAP_PROP_AUTOFOCUS))

    def trigger_oneshot_af(self, callback=None):
        """1.5초 동안 오토포커스로 초점을 맞춘 뒤, 즉시 오토포커스를 끄고 그 위치에 고정(Lock)."""
        def _worker():
            print(f"  🔍 [/dev/video{self.device_id}] 원샷 오토포커스 조절 시작...")
            self.set_autofocus(True)
            time.sleep(1.5)
            # 현재 잡힌 최적의 초점 값 읽기
            cur_f = int(self.get_focus())
            self.set_autofocus(False)
            self.set_focus(cur_f)
            print(f"  🔒 [/dev/video{self.device_id}] 초점 고정 완료: FOCUS = {cur_f}")
            if callback:
                callback(self.device_id, cur_f)

        threading.Thread(target=_worker, daemon=True).start()

    def read(self) -> Optional[np.ndarray]:
        if not self.is_opened or self.cap is None:
            return None
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return frame
        return None

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_opened = False


def draw_hud(
    frame: np.ndarray,
    title: str,
    device_id: int,
    color_bgr: Tuple[int, int, int],
    fps: float,
    cur_focus: float,
    is_autofocus: bool,
    show_grid: bool = True,
) -> np.ndarray:
    """화면에 좌/우 구분 타이틀, 박스, 십자선, 수평 정렬선 및 초점 상태를 오버레이합니다."""
    h, w = frame.shape[:2]
    vis = frame.copy()

    # 1. 외곽 테두리 강조선
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), color_bgr, 3)

    # 2. 상단 라벨 헤더 배경 바
    cv2.rectangle(vis, (0, 0), (w, 42), color_bgr, -1)
    # 타이틀
    cv2.putText(
        vis,
        f"{title} (/dev/video{device_id})",
        (12, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 3. 우측 정보: FPS 및 초점(Focus) 상태
    af_str = "AUTO" if is_autofocus else f"LOCK:{int(cur_focus)}"
    af_color = (0, 255, 255) if is_autofocus else (50, 255, 50)  # AUTO=노랑, LOCK=연두
    info_text = f"{w}x{h} @ {fps:.0f}FPS | AF:{af_str}"
    
    (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(vis, (w - tw - 16, 8), (w - 8, 34), (0, 0, 0), -1)
    cv2.putText(
        vis,
        info_text,
        (w - tw - 12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        af_color,
        1,
        cv2.LINE_AA,
    )

    # 4. 수평 정렬 가이드라인 (양안 수평선 및 중앙 십자선)
    if show_grid:
        cx, cy = w // 2, h // 2
        cv2.line(vis, (cx, 0), (cx, h), (100, 100, 100), 1, cv2.LINE_AA)
        
        # 3등분 수평 가이드선 (스테레오 수평 일치 점검용)
        for y_pos in [int(h * 0.33), cy, int(h * 0.66)]:
            line_color = (0, 255, 255) if y_pos == cy else (180, 180, 0)
            line_thickness = 2 if y_pos == cy else 1
            cv2.line(vis, (0, y_pos), (w, y_pos), line_color, line_thickness, cv2.LINE_AA)

        # 중앙 십자 포인트
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

    return vis


def run_dual_eyes_checker(
    left_id: int,
    right_id: int,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    disable_autofocus: bool = True,
    initial_focus: int = 25,
):
    available_cams = list_available_cameras()
    print("\n" + "=" * 68)
    print("🔍 [시스템 감지된 비디오 디바이스 목록]")
    for idx, name in available_cams:
        mark = ""
        if idx == left_id:
            mark = " 👈 [현재 LEFT]"
        elif idx == right_id:
            mark = " 👉 [현재 RIGHT]"
        print(f"  /dev/video{idx:2d} : {name}{mark}")
    print("=" * 68 + "\n")

    print(f"🚀 양안 카메라 연결: LEFT=/dev/video{left_id}, RIGHT=/dev/video{right_id}")
    print(f"  해상도: {width}x{height}, FPS: {fps}")
    print(f"  🔒 초점 설정: 오토포커스 해제={disable_autofocus}, 초기 고정 초점값={initial_focus}")

    cam_left = CameraStream(
        left_id, width=width, height=height, fps=fps,
        disable_autofocus=disable_autofocus, initial_focus=initial_focus
    )
    cam_right = CameraStream(
        right_id, width=width, height=height, fps=fps,
        disable_autofocus=disable_autofocus, initial_focus=initial_focus
    )

    has_display = bool(os.environ.get("DISPLAY"))
    window_name = "Dual Eyes Stereo View: LEFT vs RIGHT (Press 'f' to AF-Lock, 's' to swap, 'q' to quit)"
    if has_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "-" * 68)
    print("🎮 [키보드 단축키 안내]")
    print("  'f' : 🔍 원샷 오토포커스 후 자동 고정 (One-Shot AF & Lock)")
    print("  'a' : 오토포커스 On/Off 토글 (Auto ↔ Lock)")
    print("  '+' / ']' : 초점 거리 증가 (+5)")
    print("  '-' / '[' : 초점 거리 감소 (-5)")
    print("  's' : 왼쪽/오른쪽 카메라 즉시 뒤바꿈 (Swap Left <-> Right)")
    print("  'g' : 수평선 및 중앙 십자선 가이드 On/Off")
    print("  'c' : 현재 프레임 스냅샷 저장")
    print("  'q' or ESC : 프로그램 종료")
    print("-" * 68 + "\n")

    show_grid = True
    fps_left, fps_right = 0.0, 0.0
    frame_count = 0
    fps_timer = time.time()

    save_dir = "logs/camera_check"
    os.makedirs(save_dir, exist_ok=True)

    try:
        while True:
            frame_l = cam_left.read()
            frame_r = cam_right.read()

            frame_count += 1
            if time.time() - fps_timer >= 0.5:
                dur = time.time() - fps_timer
                fps_calc = frame_count / dur
                fps_left = fps_calc if frame_l is not None else 0.0
                fps_right = fps_calc if frame_r is not None else 0.0
                frame_count = 0
                fps_timer = time.time()

            # 신호 없음 대체 화면 (더미 블랙 프레임)
            if frame_l is None:
                frame_l = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame_l, f"NO SIGNAL: /dev/video{left_id}", (width // 6, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if frame_r is None:
                frame_r = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame_r, f"NO SIGNAL: /dev/video{right_id}", (width // 6, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 두 카메라의 출력 해상도가 다를 경우 통일 리사이즈
            if frame_l.shape[:2] != (height, width):
                frame_l = cv2.resize(frame_l, (width, height))
            if frame_r.shape[:2] != (height, width):
                frame_r = cv2.resize(frame_r, (width, height))

            # 초점 및 AF 상태 획득
            f_val_l = cam_left.get_focus()
            af_l = cam_left.get_autofocus()
            f_val_r = cam_right.get_focus()
            af_r = cam_right.get_autofocus()

            # 좌안(초록) & 우안(주황) HUD 렌더링
            vis_l = draw_hud(frame_l, "◀ LEFT EYE", left_id, (50, 180, 50), fps_left, f_val_l, af_l, show_grid=show_grid)
            vis_r = draw_hud(frame_r, "RIGHT EYE ▶", right_id, (30, 120, 255), fps_right, f_val_r, af_r, show_grid=show_grid)

            # 두 화면 나란히 결합
            combined = np.hstack([vis_l, vis_r])

            # 하단 통합 컨트롤 바
            bar_h = 32
            ctrl_bar = np.full((bar_h, combined.shape[1], 3), 30, dtype=np.uint8)
            tip_msg = "[f]: AF-Lock | [a]: AF Toggle | [+/-]: Focus Adj | [s]: Swap | [g]: Grid | [c]: Cap | [q]: Quit"
            cv2.putText(ctrl_bar, tip_msg, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            combined_view = np.vstack([combined, ctrl_bar])

            if has_display:
                cv2.imshow(window_name, combined_view)
                key = cv2.waitKey(1) & 0xFF
            else:
                if frame_count == 0:
                    preview_path = os.path.join(save_dir, "dual_eyes_preview.jpg")
                    cv2.imwrite(preview_path, combined_view)
                    print(f"\r[Headless] L(vid{left_id}) Focus={f_val_l:.0f} | R(vid{right_id}) Focus={f_val_r:.0f}", end="")
                time.sleep(0.03)
                key = 255

            if key in [ord("q"), 27]:  # 'q' or ESC
                print("\n종료 요청을 받았습니다.")
                break

            elif key == ord("f"):  # One-shot Auto Focus and Lock
                print("\n🔍 양안 원샷 오토포커스 실행 중... (초점 맞춘 후 자동 고정)")
                cam_left.trigger_oneshot_af()
                cam_right.trigger_oneshot_af()

            elif key == ord("a"):  # Toggle AF
                cur_af = cam_left.get_autofocus()
                new_af = not cur_af
                cam_left.set_autofocus(new_af)
                cam_right.set_autofocus(new_af)
                print(f"\n⚡ 오토포커스 모드 변경: {'AUTO' if new_af else 'MANUAL (LOCKED)'}")

            elif key in [ord("+"), ord("="), ord("]")]:  # Focus +5
                cur_f = int(cam_left.get_focus())
                new_f = min(255, cur_f + 5)
                cam_left.set_focus(new_f)
                cam_right.set_focus(new_f)
                print(f"\n🔍 초점 증가 (+5) -> FOCUS = {new_f}")

            elif key in [ord("-"), ord("_"), ord("[")]:  # Focus -5
                cur_f = int(cam_left.get_focus())
                new_f = max(0, cur_f - 5)
                cam_left.set_focus(new_f)
                cam_right.set_focus(new_f)
                print(f"\n🔍 초점 감소 (-5) -> FOCUS = {new_f}")

            elif key == ord("s"):  # Swap Left <-> Right
                print(f"\n🔄 [스왑] 좌/우 카메라 교체: Left({left_id}->{right_id}), Right({right_id}->{left_id})")
                cam_left.release()
                cam_right.release()
                left_id, right_id = right_id, left_id
                cam_left = CameraStream(
                    left_id, width=width, height=height, fps=fps,
                    disable_autofocus=disable_autofocus, initial_focus=initial_focus
                )
                cam_right = CameraStream(
                    right_id, width=width, height=height, fps=fps,
                    disable_autofocus=disable_autofocus, initial_focus=initial_focus
                )

            elif key == ord("g"):  # Toggle Grid
                show_grid = not show_grid
                print(f"\n📏 수평/십자 가이드라인: {'ON' if show_grid else 'OFF'}")

            elif key in [ord("c"), ord(" ")]:  # Capture
                ts = time.strftime("%Y%m%d_%H%M%S")
                l_path = os.path.join(save_dir, f"left_{left_id}_{ts}.jpg")
                r_path = os.path.join(save_dir, f"right_{right_id}_{ts}.jpg")
                c_path = os.path.join(save_dir, f"dual_{ts}.jpg")
                cv2.imwrite(l_path, frame_l)
                cv2.imwrite(r_path, frame_r)
                cv2.imwrite(c_path, combined_view)
                print(f"\n📸 [스냅샷 저장 완료] Focus: L={f_val_l}, R={f_val_r}")
                print(f"  - 듀얼: {c_path}")

    finally:
        cam_left.release()
        cam_right.release()
        if has_display:
            cv2.destroyAllWindows()
        print("\n카메라 리소스 해제 완료. 정상 종료되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="양안(Dual Eyes) 카메라 시점 및 초점 고정(Auto Focus Lock) 점검 도구")
    parser.add_argument("--left_id", type=int, default=4, help="왼쪽 카메라 /dev/video 번호 (기본값: 4)")
    parser.add_argument("--right_id", type=int, default=6, help="오른쪽 카메라 /dev/video 번호 (기본값: 6)")
    parser.add_argument("--width", type=int, default=640, help="카메라 가로 해상도 (기본값: 640)")
    parser.add_argument("--height", type=int, default=480, help="카메라 세로 해상도 (기본값: 480)")
    parser.add_argument("--fps", type=int, default=30, help="카메라 FPS (기본값: 30)")
    parser.add_argument("--enable_autofocus", action="store_true", help="오토포커스를 켠 상태로 실행 (기본은 고정)")
    parser.add_argument("--focus", type=int, default=25, help="수동 고정 초점 값 (0~255, 기본값: 25)")

    args = parser.parse_args()
    run_dual_eyes_checker(
        left_id=args.left_id,
        right_id=args.right_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        disable_autofocus=not args.enable_autofocus,
        initial_focus=args.focus,
    )
