"""손목 모터 ID 4번 / 5번 실시간 확인 스크립트.

Follower 로봇의 4번, 5번 모터 토크를 끄고, 사람이 손으로 움직였을 때
어느 모터 ID의 Raw 값이 변하는지 실시간으로 출력합니다.
"""

import sys
import os
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hardware.sts3215_driver import STS3215Driver

def main():
    port = "/dev/ttyACM0"
    driver = STS3215Driver(port, baudrate=1000000)
    if not driver.connect():
        print(f"❌ 포트 {port} 연결 실패!")
        return

    try:
        # 4번, 5번 모터 토크 해제
        print("\n" + "=" * 65)
        print(" 🔍 [손목 4번 / 5번 모터 ID 판별 도구]")
        print("  - 4번, 5번 모터의 토크를 해제했습니다.")
        print("  - 손으로 [손목 상하(Pitch)]와 [손목 회전(Roll)]을 각각 움직여보세요.")
        print("  - 종료하려면 [Ctrl + C]를 누르세요.")
        print("=" * 65 + "\n")
        
        driver.set_torque(4, False)
        driver.set_torque(5, False)

        while True:
            p4 = driver.read_position(4)
            p5 = driver.read_position(5)
            sys.stdout.write(f"\r  ▶ Motor ID 4: {str(p4):>5}  |  Motor ID 5: {str(p5):>5}     ")
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n종료합니다.")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
