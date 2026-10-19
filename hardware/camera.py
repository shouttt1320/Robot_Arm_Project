import cv2
import argparse
import time
import os
import glob


def list_camera_devices():
    """시스템에 연결된 비디오 디바이스 목록과 이름을 출력합니다."""
    print("=" * 50)
    print("시스템 비디오 디바이스 목록:")
    video_devices = sorted(glob.glob('/sys/class/video4linux/video*'))
    
    available_devices = []
    for dev_path in video_devices:
        dev_name = os.path.basename(dev_path)
        idx = dev_name.replace('video', '')
        try:
            with open(os.path.join(dev_path, 'name'), 'r') as f:
                name = f.read().strip()
            print(f" - /dev/{dev_name} (Index: {idx}) : {name}")
            available_devices.append(int(idx))
        except Exception:
            pass
    print("=" * 50)
    return available_devices


def test_camera(device_id=4, width=640, height=480, fps=30):
    print(f"\n[카메라 연결 시도] Device Index: {device_id} ({width}x{height} @ {fps}fps)")
    
    # V4L2 백엔드로 카메라 열기
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"[오류] /dev/video{device_id} 를 열 수 없습니다. 다른 인덱스를 시도해보세요.")
        return

    # 해상도 및 FPS 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[성공] 카메라 열림! 실제 설정 해상도: {actual_w}x{actual_h}, FPS: {actual_fps}")
    print("\n[단축키]")
    print(" - 'q' 또는 ESC : 프로그램 종료")
    print(" - 's' : 현재 화면 이미지로 저장\n")

    window_name = f"RGB Camera Test (Device {device_id})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    curr_fps = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[경고] 프레임을 읽어오지 못했습니다.")
                time.sleep(0.1)
                continue

            # 실시간 FPS 계산
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                curr_fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            # 화면에 정보 오버레이 (해상도, FPS)
            info_text = f"Dev:{device_id} | {actual_w}x{actual_h} | FPS: {curr_fps:.1f}"
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                print("[종료] 사용자에 의해 종료되었습니다.")
                break
            elif key == ord('s'):  # 캡처 저장
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = f"camera_capture_{timestamp}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"[저장 완료] 현재 프레임이 저장되었습니다: {save_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RGB 카메라 실시간 테스트 스크립트")
    parser.add_argument(
        '--device',
        type=int,
        default=4,
        help="카메라 디바이스 인덱스 (/dev/videoX의 X 번호, 기본값: 4)",
    )
    parser.add_argument('--width', type=int, default=640, help="카메라 가로 해상도 (기본값: 640)")
    parser.add_argument('--height', type=int, default=480, help="카메라 세로 해상도 (기본값: 480)")
    parser.add_argument('--fps', type=int, default=30, help="카메라 FPS (기본값: 30)")
    parser.add_argument(
        '--list',
        action='store_true',
        help="사용 가능한 카메라 디바이스 목록만 출력하고 종료",
    )

    args = parser.parse_args()

    # 장치 목록 표시
    list_camera_devices()

    if not args.list:
        test_camera(
            device_id=args.device,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
