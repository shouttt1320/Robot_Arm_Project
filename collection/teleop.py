"""SO-101 리더-팔로워 실시간 텔레오퍼레이션 (Phase 3).

calibration.json 에 기록된 영점과 범위를 바탕으로,
Leader 팔을 손으로 조작하면 Follower 팔이 실시간(50Hz)으로 1:1 정밀 추종합니다.
"""

import argparse
import json
import os
import signal
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


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


class TeleoperationController:
    def __init__(self, config_path: str = "configs/calibration.json", target_fps: int = 50):
        if not os.path.exists(config_path):
            alt_path = "calibration.json" if config_path != "calibration.json" else "configs/calibration.json"
            if os.path.exists(alt_path):
                config_path = alt_path
            else:
                raise FileNotFoundError(f"캘리브레이션 파일({config_path})을 찾을 수 없습니다! 먼저 calibrate.py를 실행하세요.")

        self.config_path = config_path

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.leader_port = self.config["leader_port"]
        self.follower_port = self.config["follower_port"]
        self.joints_cfg = self.config["joints"]
        self.gripper_cfg = self.config["gripper"]

        self.l_driver = STS3215Driver(self.leader_port, baudrate=1000000, timeout=0.03)
        self.f_driver = STS3215Driver(self.follower_port, baudrate=1000000, timeout=0.03)

    def connect(self):
        print(f"\n[통신 연결]")
        print(f" - Leader 로봇   : {self.leader_port}")
        print(f" - Follower 로봇 : {self.follower_port}")

        if not self.l_driver.connect():
            raise ConnectionError(f"Leader 포트({self.leader_port}) 연결 실패")
        if not self.f_driver.connect():
            self.l_driver.close()
            raise ConnectionError(f"Follower 포트({self.follower_port}) 연결 실패")

        # 리더는 토크 OFF (자유롭게 조작), 팔로워는 토크 ON (추종)
        print("\n[모터 토크 설정]")
        print(" - Leader 팔   : 토크 OFF (손으로 자유롭게 조작 가능)")
        self.l_driver.set_all_torques(range(1, 7), False)
        
        print(" - Follower 팔 : 토크 ON (추종 대기)")
        self.f_driver.set_all_torques(range(1, 7), True)
        time.sleep(0.5)

    def map_leader_to_follower(self, leader_positions: dict) -> dict:
        """리더 팔의 관절 위치를 캘리브레이션 기준에 따라 팔로워 목표 각도로 변환"""
        follower_goals = {}

        # 1번 ~ 5번 관절 매핑 (영점 오프셋 + 회전 방향)
        for i in range(1, 6):
            key = f"joint_{i}"
            cfg = self.joints_cfg[key]
            l_val = leader_positions.get(i)
            if l_val is None:
                continue

            l_zero = cfg["leader_zero"]
            f_zero = cfg["follower_zero"]
            direction = cfg.get("direction", 1)
            min_lim = cfg.get("follower_min", cfg.get("min", 0))
            max_lim = cfg.get("follower_max", cfg.get("max", 4095))

            # 상대적 변화량에 방향 부호를 곱해 팔로워 목표값 계산
            delta = (l_val - l_zero) * direction
            f_target = f_zero + delta

            # 안전 범위 제한
            follower_goals[i] = int(clamp(f_target, min_lim, max_lim))

        # 6번 그리퍼 매핑 (개폐 비율 기준 매핑)
        l_grip = leader_positions.get(6)
        if l_grip is not None:
            l_open = self.gripper_cfg["leader_open"]
            l_close = self.gripper_cfg["leader_close"]
            f_open = self.gripper_cfg["follower_open"]
            f_close = self.gripper_cfg["follower_close"]

            # 리더 개폐 비율 (0.0=완전닫힘 ~ 1.0=완전열림)
            denom = l_open - l_close
            if denom != 0:
                ratio = (l_grip - l_close) / denom
            else:
                ratio = 0.5
            ratio = clamp(ratio, 0.0, 1.0)

            # 팔로워 그리퍼 목표치 계산
            f_grip_target = f_close + ratio * (f_open - f_close)
            
            # 그리퍼 실제 가동 범위(close ~ open) 기준으로 안전 클램핑
            grip_min = min(f_open, f_close)
            grip_max = max(f_open, f_close)
            follower_goals[6] = int(clamp(f_grip_target, grip_min, grip_max))

        return follower_goals

    def run(self):
        self.connect()
        self.running = True

        print("\n" + "=" * 65)
        print(" 🎮 실시간 텔레오퍼레이션 시작! (Leader ➔ Follower)")
        print(f" 제어 주기: {self.target_fps} Hz (종료: Ctrl + C)")
        print("=" * 65)

        prev_time = time.time()
        fps_timer = time.time()
        loop_count = 0
        current_fps = 0.0

        try:
            while self.running:
                loop_start = time.time()

                # 1. Leader 팔의 현재 관절 위치(1~6) 읽기
                leader_pos = self.l_driver.read_positions(range(1, 7))

                # 모든 모터가 정상적으로 읽혔는지 확인
                valid_read = all(leader_pos.get(i) is not None for i in range(1, 7))

                if valid_read:
                    # 2. 리더 각도를 팔로워 목표 각도로 변환
                    follower_goals = self.map_leader_to_follower(leader_pos)

                    # 3. 팔로워 로봇에 6개 모터 목표 위치 동시 전송 (Sync Write)
                    self.f_driver.sync_write_positions(follower_goals, speed=0, acc=0)

                # FPS 계산
                loop_count += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    current_fps = loop_count / (now - fps_timer)
                    loop_count = 0
                    fps_timer = now

                    # 상태 표시 (리더와 팔로워의 현재 목표값)
                    if valid_read:
                        status = f"FPS: {current_fps:4.1f} | J1:{follower_goals[1]:>4} J2:{follower_goals[2]:>4} J3:{follower_goals[3]:>4} J4:{follower_goals[4]:>4} J5:{follower_goals[5]:>4} Grip:{follower_goals[6]:>4}"
                        sys.stdout.write("\r" + status)
                        sys.stdout.flush()

                # 50Hz 주기 유지 (Sleep)
                elapsed = time.time() - loop_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self.stop()

    def stop(self):
        if hasattr(self, "_stopped") and self._stopped:
            return
        self._stopped = True
        self.running = False

        print("\n\n[안전 정지] Follower 및 Leader의 모든 모터 토크를 안전하게 해제합니다...")
        try:
            # 1. Follower 모터 토크 일괄 해제 (브로드캐스트 + 개별 확인사살)
            if self.f_driver and self.f_driver.ser and self.f_driver.ser.is_open:
                self.f_driver.set_all_torques(range(1, 7), False)
            
            # 2. Leader 모터 토크 일괄 해제
            if self.l_driver and self.l_driver.ser and self.l_driver.ser.is_open:
                self.l_driver.set_all_torques(range(1, 7), False)
            
            time.sleep(0.05)
        except Exception as err:
            print(f"[경고] 토크 해제 중 예외 발생: {err}")
        finally:
            if self.l_driver:
                self.l_driver.close()
            if self.f_driver:
                self.f_driver.close()
        print("✅ 6개 모든 모터의 토크가 정상 해제되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 리더-팔로워 실시간 텔레오퍼레이션")
    parser.add_argument(
        "--config",
        type=str,
        default="calibration.json",
        help="캘리브레이션 JSON 파일 경로 (기본값: calibration.json)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="제어 주기 FPS (기본값: 50Hz)",
    )
    args = parser.parse_args()

    controller = TeleoperationController(config_path=args.config, target_fps=args.fps)

    # Ctrl+C 시 안전 종료 시그널 핸들러
    def sigint_handler(sig, frame):
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    controller.run()


if __name__ == "__main__":
    main()
