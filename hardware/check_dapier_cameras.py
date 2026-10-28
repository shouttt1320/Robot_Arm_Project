"""SO-101 DAPIER-ACT 카메라 사전 점검 도구 (OS30A Depth + Dual Wrist RGB).

점검 항목:
1. OS30A 뎁스 카메라: 640x460 뎁스 히트맵 + 512점 3D 포인트클라우드 Bird's Eye View(탑뷰) 투영 확인
2. Left Wrist 카메라: 240x320 RGB 영상 및 FPS 점검
3. Right Wrist 카메라: 240x320 RGB 영상 및 FPS 점검
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 프로젝트 루트 경로 등록
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.camera_interface import MultiCameraCapture
from hardware.os30a_camera import OS30ACamera
from models.so101_dapier_act.config import DapierACTConfig


def render_pointcloud_bev(
    points: np.ndarray,
    crop_x=(0.15, 0.32),
    crop_y=(-0.26, 0.26),
    canvas_size=(240, 320),
) -> np.ndarray:
    """3D 포인트클라우드([N, 3])를 2D Bird's-Eye-View (탑뷰 평면) 캔버스에 렌더링"""
    h, w = canvas_size
    canvas = np.full((h, w, 3), 25, dtype=np.uint8)  # 다크 그레이 배경

    # 작업영역 가이드라인 사각형
    cv2.rectangle(canvas, (10, 10), (w - 10, h - 10), (60, 60, 60), 1)

    if len(points) == 0:
        cv2.putText(canvas, "No Valid Points", (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas

    # X, Y를 픽셀 좌표로 투영 (X: 전방, Y: 좌우)
    x_min, x_max = crop_x
    y_min, y_max = crop_y

    px = np.clip((points[:, 1] - y_min) / (y_max - y_min + 1e-6) * (w - 20) + 10, 0, w - 1).astype(int)
    py = np.clip((x_max - points[:, 0]) / (x_max - x_min + 1e-6) * (h - 20) + 10, 0, h - 1).astype(int)

    # 높이(Z)에 따른 색상 그라데이션 (파란색=바닥, 빨간색=높음)
    z = points[:, 2]
    z_norm = np.clip((z - 0.005) / 0.25, 0.0, 1.0)

    for i in range(len(points)):
        val = int(z_norm[i] * 255)
        color = (255 - val, val // 2, val)  # BGR
        cv2.circle(canvas, (px[i], py[i]), 2, color, -1)

    # 안내 텍스트
    cv2.putText(canvas, f"DP3 PointCloud: {len(points)} pts", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, "Top (Robot Forward)", (w // 2 - 60, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="DAPIER-ACT 멀티 카메라 사전 점검기")
    parser.add_argument("--left_wrist_cam", type=int, default=4, help="Left Wrist 카메라 V4L2 ID (기본값: 4)")
    parser.add_argument("--right_wrist_cam", type=int, default=6, help="Right Wrist 카메라 V4L2 ID (기본값: 6)")
    parser.add_argument("--os30a_id", type=int, default=None, help="OS30A 디바이스 ID (생략 시 샘플/자동)")
    args = parser.parse_args()

    cfg = DapierACTConfig()

    print("=" * 70)
    print(" 🔍 [SO-101 DAPIER-ACT 카메라 통합 점검 도구]")
    print(f" • Left Wrist 카메라  : /dev/video{args.left_wrist_cam}")
    print(f" • Right Wrist 카메라 : /dev/video{args.right_wrist_cam}")
    print(f" • OS30A 뎁스 카메라  : ID={args.os30a_id}")
    print("=" * 70)

    # 1. OS30A 초기화
    os30a = OS30ACamera(config=cfg, device_index=args.os30a_id, use_sample_fallback=True)

    # 2. 손목 카메라 초기화
    cam_dict = {"left_wrist": args.left_wrist_cam, "right_wrist": args.right_wrist_cam}
    wrist_cams = MultiCameraCapture(cam_dict, width=cfg.wrist_width, height=cfg.wrist_height, fps=30)

    print("\n🖥️ 점검 창이 실행되었습니다. 종료하려면 점검 창에서 [Q] 또는 [ESC]를 누르세요.\n")

    fps_count = 0
    fps_start = time.time()
    current_fps = 0.0

    try:
        while True:
            t0 = time.time()

            # 1. 손목 카메라 프레임 읽기
            wrist_frames = wrist_cams.read_all()
            left_img = wrist_frames.get("left_wrist")
            right_img = wrist_frames.get("right_wrist")

            if left_img is None:
                left_img = np.zeros((cfg.wrist_height, cfg.wrist_width, 3), dtype=np.uint8)
                cv2.putText(left_img, "Left Wrist Disconnected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if right_img is None:
                right_img = np.zeros((cfg.wrist_height, cfg.wrist_width, 3), dtype=np.uint8)
                cv2.putText(right_img, "Right Wrist Disconnected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 2. OS30A 뎁스 및 포인트클라우드 읽기
            ret_d, depth_raw = os30a.read_depth()
            ret_p, pcd_pts = os30a.read_pointcloud()

            # 뎁스 컬러맵 시각화 (0~1000mm 범위 매핑)
            depth_scaled = np.clip(depth_raw / 1000.0 * 255.0, 0, 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
            depth_resized = cv2.resize(depth_colormap, (cfg.wrist_width, cfg.wrist_height), interpolation=cv2.INTER_LINEAR)

            # 포인트클라우드 2D BEV 투영 렌더링
            bev_canvas = render_pointcloud_bev(
                pcd_pts, crop_x=cfg.crop_x, crop_y=cfg.crop_y, canvas_size=(cfg.wrist_height, cfg.wrist_width)
            )

            # 3. 2x2 대시보드 그리드 생성
            # 상단: [Left Wrist RGB] [Right Wrist RGB]
            # 하단: [OS30A Depth Heatmap] [DP3 PointCloud BEV]
            top_row = np.hstack([left_img, right_img])
            bottom_row = np.hstack([depth_resized, bev_canvas])
            dashboard = np.vstack([top_row, bottom_row])

            # FPS 연산
            fps_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_count / (time.time() - fps_start)
                fps_count = 0
                fps_start = time.time()

            # 대시보드 상단 정보 오버레이
            cv2.putText(dashboard, "LEFT WRIST RGB", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(dashboard, "RIGHT WRIST RGB", (cfg.wrist_width + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(dashboard, "HEAD OS30A DEPTH", (10, cfg.wrist_height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(dashboard, f"FPS: {current_fps:.1f}", (cfg.wrist_width * 2 - 100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("DAPIER-ACT Multi-Camera Diagnostic Dashboard", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                break

    finally:
        print("\n🧹 카메라 연결 해제 중...")
        os30a.release()
        wrist_cams.stop()
        cv2.destroyAllWindows()
        print("✅ 카메라 점검 완료.")


if __name__ == "__main__":
    main()
