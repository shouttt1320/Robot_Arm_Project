"""SO-101 (SO-ARM100) 대화형 캘리브레이션 도구.

Leader 팔과 Follower 팔의:
1. 영점 위치 (Zero Offset)
2. 관절 회전 방향 부호 (Sign: +1 또는 -1)
3. 가동 범위 한계 (Min / Max)
4. 그리퍼 개폐 범위 (Open / Close)
를 순차적으로 측정하여 calibration.json 파일로 저장합니다.
"""

import argparse
import json
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

JOINT_NAMES = [
    "1번 Base (베이스 회전)",
    "2번 Shoulder (어깨)",
    "3번 Elbow (팔꿈치)",
    "4번 Wrist Pitch (손목 상하)",
    "5번 Wrist Roll (손목 회전)",
    "6번 Gripper (그리퍼)",
]


def print_header(title):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def get_driver_positions(driver, servo_ids=range(1, 7)):
    """모터들의 현재 위치(0~4095)를 딕셔너리로 반환"""
    pos_dict = {}
    for sid in servo_ids:
        pos = driver.read_position(sid)
        pos_dict[sid] = pos
    return pos_dict


def live_monitor_two_arms(leader, follower, prompt="엔터(Enter)를 누르면 이 자세를 기록합니다..."):
    """두 로봇팔의 현재 각도를 실시간으로 출력하며 엔터 대기"""
    print(f"\n{prompt}")
    print("(터미널에 실시간 관절 각도가 표시됩니다. 원하는 자세에 놓고 엔터를 누르세요)")
    print("-" * 65)
    print(f"{'관절':^20} | {'Leader 팔 (Raw)':^18} | {'Follower 팔 (Raw)':^18}")
    print("-" * 65)

    import select

    while True:
        l_pos = get_driver_positions(leader)
        f_pos = get_driver_positions(follower)

        status_line = ""
        for i in range(1, 7):
            lp = str(l_pos.get(i, "Err"))
            fp = str(f_pos.get(i, "Err"))
            status_line += f" J{i}: L={lp:>4} F={fp:>4} |"

        # 터미널 한 줄 갱신
        sys.stdout.write("\r" + status_line)
        sys.stdout.flush()

        # 엔터 입력 감지 (Non-blocking)
        if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
            sys.stdin.readline()
            print("\n")
            return l_pos, f_pos


def calibrate(leader_port: str, follower_port: str, save_path: str = "configs/calibration.json"):
    print_header("SO-101 리더-팔로워 캘리브레이션 (Phase 2)")
    print(f"Leader 포트  : {leader_port}")
    print(f"Follower 포트: {follower_port}")

    # 디렉토리 자동 생성
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    l_driver = STS3215Driver(leader_port, baudrate=1000000)
    f_driver = STS3215Driver(follower_port, baudrate=1000000)

    if not l_driver.connect():
        print(f"❌ Leader 포트 ({leader_port}) 연결 실패!")
        return
    if not f_driver.connect():
        print(f"❌ Follower 포트 ({follower_port}) 연결 실패!")
        l_driver.close()
        return

    try:
        # 안전을 위해 양쪽 팔 모두 토크 해제 (사람이 손으로 자유롭게 움직일 수 있는 상태)
        print("\n[안내] 두 로봇팔의 모터 토크를 모두 해제(Torque Off)합니다.")
        l_driver.set_all_torques(range(1, 7), False)
        f_driver.set_all_torques(range(1, 7), False)
        time.sleep(0.5)

        calib_data = {
            "leader_port": leader_port,
            "follower_port": follower_port,
            "num_joints": 6,
            "joints": {},
        }

        # ---------------------------------------------------------------------
        # [Step 1] 영점 (Home / Zero) 위치 설정
        # ---------------------------------------------------------------------
        print_header("Step 1: 홈 포지션 (영점 / Zero Offset) 설정")
        print("리더 팔과 팔로워 팔을 '완전히 동일한 기준 자세'로 손으로 맞추어 주세요.")
        print("(예: 팔을 일직선으로 곧게 펴거나 수직으로 세우고, 그리퍼를 닫은 상태 등)")
        
        l_zero, f_zero = live_monitor_two_arms(
            l_driver, f_driver,
            prompt="👉 두 로봇팔의 자세가 일치하면 [Enter] 키를 누르세요."
        )

        for i in range(1, 7):
            calib_data["joints"][f"joint_{i}"] = {
                "name": JOINT_NAMES[i-1],
                "leader_zero": l_zero[i],
                "follower_zero": f_zero[i],
                "direction": 1,  # 기본값 +1
                "min": 0,
                "max": 4095,
            }

        print("✅ 홈 포지션 영점 기록 완료!")
        for i in range(1, 7):
            print(f"  - {JOINT_NAMES[i-1]}: Leader Zero={l_zero[i]}, Follower Zero={f_zero[i]}")

        # ---------------------------------------------------------------------
        # [Step 2] 회전 방향 (Sign: +1 또는 -1) 매핑 확인
        # ---------------------------------------------------------------------
        print_header("Step 2: 회전 방향 (Direction / Sign) 확인")
        print("리더 팔과 팔로워 팔의 모터 조립 방향이 같으면 [+1], 반대면 [-1]입니다.")
        print("보통 SO-101 리더-팔로워는 모든 모터가 같은 방향(+1)으로 조립됩니다.")
        
        dir_choice = input("\n모든 관절의 방향 부호를 기본값(+1)으로 설정할까요? (Y/n): ").strip().lower()
        if dir_choice in ["", "y", "yes"]:
            for i in range(1, 7):
                calib_data["joints"][f"joint_{i}"]["direction"] = 1
            print("✅ 모든 관절 방향: +1 (동일 방향) 설정 완료")
        else:
            print("각 관절별로 방향을 설정합니다 (+1 또는 -1):")
            for i in range(1, 7):
                while True:
                    val = input(f" - {JOINT_NAMES[i-1]} 방향 (+1 / -1) [기본 +1]: ").strip()
                    if val in ["", "+1", "1"]:
                        calib_data["joints"][f"joint_{i}"]["direction"] = 1
                        break
                    elif val == "-1":
                        calib_data["joints"][f"joint_{i}"]["direction"] = -1
                        break
                    print("  ⚠️ +1 또는 -1 을 입력해 주세요.")

        # ---------------------------------------------------------------------
        # [Step 3] Follower 팔 안전 가동 범위 (Min / Max) 측정
        # ---------------------------------------------------------------------
        print_header("Step 3: Follower 팔(실제 동작 로봇) 안전 가동 범위 설정")
        print("💡 [중요] 실제 모터 힘으로 움직이는 'Follower 팔'이 책상이나 기구물에")
        print("   부딪히지 않도록 안전 한계선(Min/Max)을 Follower 기준으로 직접 측정합니다.")
        print("\n👉 Follower 팔을 손으로 잡고, 관절별로 안전하게 움직일 수 있는 끝까지 천천히 움직여 주세요.")
        print("(원하는 만큼 충분히 모든 관절을 움직인 후, 완료되면 [Enter] 키를 누르세요!)")
        print("-" * 75)
        
        import select
        min_pos = {i: 4095 for i in range(1, 7)}
        max_pos = {i: 0 for i in range(1, 7)}
        
        while True:
            # Follower 팔의 실제 모터 위치를 읽어 최소/최대값 갱신
            pos = get_driver_positions(f_driver)
            for i in range(1, 7):
                if pos[i] is not None:
                    min_pos[i] = min(min_pos[i], pos[i])
                    max_pos[i] = max(max_pos[i], pos[i])
            
            # 실시간으로 Follower 각 관절의 Min ~ Max와 현재값 출력
            status_parts = []
            for i in range(1, 7):
                curr = str(pos.get(i, "Err"))
                status_parts.append(f"F_J{i}:[{min_pos[i]:>4}~{max_pos[i]:>4} (현재:{curr:>4})]")
            
            sys.stdout.write("\r" + " | ".join(status_parts))
            sys.stdout.flush()

            # 엔터(Enter) 입력 감지 시 종료
            if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.readline()
                print("\n\n✅ Follower 가동 범위 측정 완료!")
                break

        for i in range(1, 7):
            f_min = max(0, min_pos[i] - 15)
            f_max = min(4095, max_pos[i] + 15)
            calib_data["joints"][f"joint_{i}"]["follower_min"] = f_min
            calib_data["joints"][f"joint_{i}"]["follower_max"] = f_max
            # 하위 호환성을 위해 min, max 키에도 Follower 기준값 저장
            calib_data["joints"][f"joint_{i}"]["min"] = f_min
            calib_data["joints"][f"joint_{i}"]["max"] = f_max
            print(f" - Follower {JOINT_NAMES[i-1]}: Min={f_min} ~ Max={f_max}")

        # ---------------------------------------------------------------------
        # [Step 4] 그리퍼 (6번 모터) 완전 열림 / 완전 닫힘 측정
        # ---------------------------------------------------------------------
        print_header("Step 4: 그리퍼 (Gripper) 열림 / 닫힘 설정")
        print("1) 두 로봇팔의 그리퍼를 손으로 '완전히 벌린(Open)' 상태로 놓아주세요.")
        l_open, f_open = live_monitor_two_arms(
            l_driver, f_driver,
            prompt="👉 그리퍼를 '완전히 연 상태'에서 [Enter]를 누르세요."
        )

        print("2) 두 로봇팔의 그리퍼를 손으로 '완전히 다문(Closed)' 상태로 놓아주세요.")
        l_close, f_close = live_monitor_two_arms(
            l_driver, f_driver,
            prompt="👉 그리퍼를 '완전히 닫은 상태'에서 [Enter]를 누르세요."
        )

        calib_data["gripper"] = {
            "leader_open": l_open[6],
            "leader_close": l_close[6],
            "follower_open": f_open[6],
            "follower_close": f_close[6],
        }
        print("✅ 그리퍼 개폐 범위 설정 완료!")
        print(f"  - Leader Gripper   : Open={l_open[6]}, Close={l_close[6]}")
        print(f"  - Follower Gripper : Open={f_open[6]}, Close={f_close[6]}")

        # ---------------------------------------------------------------------
        # 파일 저장
        # ---------------------------------------------------------------------
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=4, ensure_ascii=False)

        print_header("🎉 캘리브레이션 완료!")
        print(f"설정 파일이 저장되었습니다: {os.path.abspath(save_path)}")
        print("\n이제 다음 단계인 [Phase 3] 텔레오퍼레이션 실시간 추종 테스트를 진행할 수 있습니다.")

    finally:
        l_driver.close()
        f_driver.close()


def main():
    parser = argparse.ArgumentParser(description="SO-101 리더-팔로워 캘리브레이션 도구")
    parser.add_argument(
        "--leader",
        type=str,
        default="/dev/ttyACM1",
        help="Leader 로봇 시리얼 포트 (기본값: /dev/ttyACM1)",
    )
    parser.add_argument(
        "--follower",
        type=str,
        default="/dev/ttyACM0",
        help="Follower 로봇 시리얼 포트 (기본값: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/calibration.json",
        help="저장할 캘리브레이션 JSON 경로 (기본값: configs/calibration.json)",
    )
    args = parser.parse_args()

    calibrate(
        leader_port=args.leader,
        follower_port=args.follower,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
