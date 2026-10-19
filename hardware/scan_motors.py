"""SO-101 로봇팔 시리얼 포트 및 STS3215 서보 모터 스캔 진단 도구.

연결된 Bus Servo Board의 시리얼 포트를 검색하고,
ID 1~6번 모터의 통신 상태(Ping), 현재 각도, 전압, 온도를 점검합니다.
"""

import argparse
import glob
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from hardware.sts3215_driver import STS3215Driver
except ImportError:
    from sts3215_driver import STS3215Driver


def find_available_serial_ports():
    """시스템에 연결된 USB 시리얼 포트 목록 검색"""
    ports = sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
    return ports


def scan_port(port: str, baudrate: int = 1000000, max_id: int = 6):
    print(f"\n[{port}] 포트 스캔 시작 (Baudrate: {baudrate} bps)...")
    driver = STS3215Driver(port, baudrate=baudrate, timeout=0.04)

    if not driver.connect():
        print(f"  ❌ 포트를 열 수 없습니다. (권한 또는 사용 중 여부 확인)")
        return []

    found_motors = []
    print("-" * 65)
    print(f"{'ID':^5} | {'상태':^8} | {'현재 위치 (0~4095)':^18} | {'전압 (V)':^10} | {'온도 (°C)':^10}")
    print("-" * 65)

    try:
        for servo_id in range(1, max_id + 1):
            status = driver.read_status(servo_id)
            if status is not None:
                pos = status['position']
                volt = status['voltage']
                temp = status['temperature']
                print(f"{servo_id:^5} | {'연결됨':^8} | {pos:^18} | {volt:^10.1f} | {temp:^10}")
                found_motors.append(servo_id)
            else:
                # 빠른 ping 한 번 더 시도
                if driver.ping(servo_id):
                    print(f"{servo_id:^5} | {'Ping 성공':^8} | {'(정보 수신 실패)':^18} | {'-':^10} | {'-':^10}")
                    found_motors.append(servo_id)
                else:
                    print(f"{servo_id:^5} | {'응답 없음':^8} | {'-':^18} | {'-':^10} | {'-':^10}")
            time.sleep(0.01)
    finally:
        driver.close()

    print("-" * 65)
    print(f"  👉 결과: 총 {len(found_motors)}개 모터 발견 (발견된 ID: {found_motors})")
    return found_motors


def main():
    parser = argparse.ArgumentParser(description="SO-101 STS3215 서보 모터 스캔 유틸리티")
    parser.add_argument(
        '--port',
        type=str,
        default=None,
        help="스캔할 시리얼 포트 (예: /dev/ttyUSB0). 미지정 시 시스템의 모든 포트 자동 스캔",
    )
    parser.add_argument(
        '--baud',
        type=int,
        default=1000000,
        help="통신 속도 (기본값: 1,000,000 bps)",
    )
    parser.add_argument(
        '--max-id',
        type=int,
        default=6,
        help="스캔할 최대 모터 ID (기본값: 6, SO-101 6자유도)",
    )

    args = parser.parse_args()

    print("=" * 65)
    print(" SO-101 STS3215 모터 통신 진단 툴 (Phase 1)")
    print("=" * 65)

    available_ports = find_available_serial_ports()
    if not available_ports:
        print("\n⚠️  현재 시스템에 인식된 시리얼 장치(/dev/ttyUSB* 또는 /dev/ttyACM*)가 없습니다.")
        print("  - Bus Servo Board의 USB 케이블이 PC에 연결되어 있는지 확인해 주세요.")
        print("  - 서보 보드의 전원(12V 등)이 켜져 있는지 확인해 주세요.\n")
        return

    print(f"발견된 시리얼 포트 목록: {available_ports}")

    if args.port:
        scan_port(args.port, baudrate=args.baud, max_id=args.max_id)
    else:
        for port in available_ports:
            scan_port(port, baudrate=args.baud, max_id=args.max_id)


if __name__ == '__main__':
    main()
