"""SO-101 (SO-ARM100) 대화형 캘리브레이션 도구.

모드:
1. 전체 캘리브레이션 (리더 팔 + 팔로워 팔 동시 캘리브레이션)
2. 팔로워 단독 캘리브레이션 (--follower-only):
   - 리더 팔 연결 불필요.
   - 기존 리더 캘리브레이션 데이터는 100% 안전하게 유지하고, 팔로워 데이터만 갱신.
   - 이전 영점 대비 모터 축 오프셋 차이 실시간 분석 및 리포트 제공.
"""

import argparse
import json
import os
import shutil
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
    print("\n" + "=" * 68)
    print(f" {title}")
    print("=" * 68)


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
    print("-" * 68)

    import select

    while True:
        l_pos = get_driver_positions(leader)
        f_pos = get_driver_positions(follower)

        status_line = ""
        for i in range(1, 7):
            lp = str(l_pos.get(i, "Err"))
            fp = str(f_pos.get(i, "Err"))
            status_line += f" J{i}: L={lp:>4} F={fp:>4} |"

        sys.stdout.write("\r" + status_line)
        sys.stdout.flush()

        if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
            sys.stdin.readline()
            print("\n")
            return l_pos, f_pos


def live_monitor_follower(driver, old_zeros=None, prompt="엔터(Enter)를 누르면 이 자세를 기록합니다..."):
    """팔로워 팔 단독 모니터링 (이전 영점과의 실시간 편차 표시)"""
    print(f"\n{prompt}")
    print("(터미널에 실시간 관절 각도가 표시됩니다. 원하는 자세에 놓고 엔터를 누르세요)")
    print("-" * 68)

    import select

    while True:
        pos = get_driver_positions(driver)

        status_line = ""
        for i in range(1, 7):
            cur = pos.get(i)
            cur_str = f"{cur:>4}" if cur is not None else " Err"
            if old_zeros and i in old_zeros and cur is not None:
                diff = cur - old_zeros[i]
                diff_str = f"({diff:+4d})"
            else:
                diff_str = ""
            status_line += f" J{i}:{cur_str}{diff_str} |"

        sys.stdout.write("\r" + status_line)
        sys.stdout.flush()

        if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
            sys.stdin.readline()
            print("\n")
            return pos


def calibrate_follower_only(follower_port: str, save_path: str = "configs/calibration.json"):
    """팔로워 팔 단독 캘리브레이션 (리더 설정은 그대로 보존하고 팔로워 필드만 업데이트)"""
    print_header("🦾 SO-101 Follower(팔로워) 로봇 팔 단독 캘리브레이션")
    print(f"Follower 포트: {follower_port}")
    print(f"설정 파일    : {save_path}")

    # 기존 calibration.json 로드
    old_zeros = {}
    calib_data = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
            # 이전 영점 추출
            for i in range(1, 7):
                j_key = f"joint_{i}"
                if j_key in calib_data.get("joints", {}):
                    old_zeros[i] = calib_data["joints"][j_key].get("follower_zero", 2048)
            # 백업 파일 생성
            bak_path = save_path + ".bak"
            shutil.copy(save_path, bak_path)
            print(f"📁 기존 설정 파일 백업 완료 ➔ {bak_path}")
        except Exception as e:
            print(f"⚠️ 기존 설정 파일 로드 실패: {e}")

    f_driver = STS3215Driver(follower_port, baudrate=1000000)
    if not f_driver.connect():
        print(f"❌ Follower 포트 ({follower_port}) 연결 실패!")
        return

    try:
        print("\n[안내] Follower 팔의 모터 토크를 해제합니다. 손으로 편안하게 움직일 수 있습니다.")
        f_driver.set_all_torques(range(1, 7), False)
        time.sleep(0.5)

        # ---------------------------------------------------------------------
        # [Step 1] 팔로워 영점 (Home / Zero) 설정
        # ---------------------------------------------------------------------
        print_header("Step 1: Follower 홈 포지션 (영점 / Zero Offset) 설정")
        print("팔로워 팔을 이전과 동일한 '기준 자세(예: 팔을 똑바로 펴거나 세운 자세)'로 손으로 맞춰주세요.")
        if old_zeros:
            print("💡 괄호 안의 (+/- 숫자)는 이전 조립 대비 현재 모터 편차(Offset)를 의미합니다.")

        f_zero = live_monitor_follower(
            f_driver,
            old_zeros=old_zeros,
            prompt="👉 원하는 홈 자세에 맞추신 후 [Enter]를 누르세요."
        )

        # ---------------------------------------------------------------------
        # [Step 2] 팔로워 안전 가동 범위 (Min / Max) 측정
        # ---------------------------------------------------------------------
        print_header("Step 2: Follower 안전 가동 범위 (Min/Max) 측정")
        print("Follower 팔을 손으로 잡고, 안전하게 움직일 수 있는 한계선까지 골고루 천천히 움직여 주세요.")
        print("(충분히 움직인 후 완료되면 [Enter] 키를 누르세요)")
        print("-" * 68)

        import select
        min_pos = {i: 4095 for i in range(1, 7)}
        max_pos = {i: 0 for i in range(1, 7)}

        while True:
            pos = get_driver_positions(f_driver)
            for i in range(1, 7):
                if pos[i] is not None:
                    min_pos[i] = min(min_pos[i], pos[i])
                    max_pos[i] = max(max_pos[i], pos[i])

            status_parts = []
            for i in range(1, 7):
                curr = str(pos.get(i, "Err"))
                status_parts.append(f"J{i}:[{min_pos[i]:>4}~{max_pos[i]:>4} (현재:{curr:>4})]")

            sys.stdout.write("\r" + " | ".join(status_parts))
            sys.stdout.flush()

            if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.readline()
                print("\n\n✅ Follower 가동 범위 측정 완료!")
                break

        # ---------------------------------------------------------------------
        # [Step 3] 그리퍼 (6번 모터) 열림 / 닫힘 측정
        # ---------------------------------------------------------------------
        print_header("Step 3: Follower 그리퍼 (6번) 열림 / 닫힘 설정")
        print("1) 팔로워 그리퍼를 손으로 '완전히 벌린(Open)' 상태로 놓아주세요.")
        f_open = live_monitor_follower(f_driver, prompt="👉 그리퍼를 '완전히 연 상태'에서 [Enter]를 누르세요.")

        print("2) 팔로워 그리퍼를 손으로 '완전히 다문(Closed)' 상태로 놓아주세요.")
        f_close = live_monitor_follower(f_driver, prompt="👉 그리퍼를 '완전히 닫은 상태'에서 [Enter]를 누르세요.")

        # ---------------------------------------------------------------------
        # 데이터 병합 및 저장
        # ---------------------------------------------------------------------
        if "joints" not in calib_data:
            calib_data["joints"] = {}
        if "gripper" not in calib_data:
            calib_data["gripper"] = {}

        calib_data["follower_port"] = follower_port
        calib_data["num_joints"] = 6

        print("\n" + "=" * 68)
        print("📊 [영점 변화 및 모터 편차 분석 리포트]")
        print(f"{'관절':^20} | {'새 영점':^10} | {'이전 영점':^10} | {'편차(Offset)':^12} | {'판정':^10}")
        print("-" * 68)

        has_large_offset = False
        offsets = {}

        for i in range(1, 7):
            j_key = f"joint_{i}"
            if j_key not in calib_data["joints"]:
                calib_data["joints"][j_key] = {
                    "name": JOINT_NAMES[i-1],
                    "leader_zero": 2048,
                    "direction": 1,
                }

            new_z = f_zero[i]
            old_z = old_zeros.get(i, new_z)
            diff = new_z - old_z
            offsets[i] = diff

            f_min = max(0, min_pos[i] - 15)
            f_max = min(4095, max_pos[i] + 15)

            calib_data["joints"][j_key]["follower_zero"] = new_z
            calib_data["joints"][j_key]["follower_min"] = f_min
            calib_data["joints"][j_key]["follower_max"] = f_max
            calib_data["joints"][j_key]["min"] = f_min
            calib_data["joints"][j_key]["max"] = f_max

            if abs(diff) <= 30:
                judge = "✅ 완벽 호환"
            elif abs(diff) <= 80:
                judge = "⚠️ 약간 편차"
            else:
                judge = "❌ 톱니 틀어짐"
                has_large_offset = True

            print(f" {JOINT_NAMES[i-1]:<20} | {new_z:^10} | {old_z:^10} | {diff:+12d} | {judge:^10}")

        calib_data["gripper"]["follower_open"] = f_open[6]
        calib_data["gripper"]["follower_close"] = f_close[6]

        # JSON 저장
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=4, ensure_ascii=False)

        print("-" * 68)
        print(f"🎉 Follower 캘리브레이션 갱신 완료! ➔ {os.path.abspath(save_path)}")

        if has_large_offset:
            print("\n⚠️ [중요 알림]")
            print("  일부 관절의 편차가 큽니다(기어 톱니가 틀어짐).")
            print("  기존 53개 데이터로 학습된 모델을 그대로 쓰시려면,")
            print("  서보 혼 톱니를 다시 맞추거나, 추론 스크립트에서 오프셋 자동 보정을 켜야 합니다.")
        else:
            print("\n✅ 모든 관절의 편차가 미세하여 기존 학습 데이터로 바로 추론해도 안전합니다!")

    finally:
        f_driver.close()


def calibrate(leader_port: str, follower_port: str, save_path: str = "configs/calibration.json"):
    print_header("SO-101 리더-팔로워 전체 캘리브레이션 (Phase 2)")
    print(f"Leader 포트  : {leader_port}")
    print(f"Follower 포트: {follower_port}")

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

        # Step 1
        print_header("Step 1: 홈 포지션 (영점 / Zero Offset) 설정")
        l_zero, f_zero = live_monitor_two_arms(
            l_driver, f_driver,
            prompt="👉 두 로봇팔의 자세가 일치하면 [Enter] 키를 누르세요."
        )

        for i in range(1, 7):
            calib_data["joints"][f"joint_{i}"] = {
                "name": JOINT_NAMES[i-1],
                "leader_zero": l_zero[i],
                "follower_zero": f_zero[i],
                "direction": 1,
                "min": 0,
                "max": 4095,
            }

        # Step 2
        print_header("Step 2: 회전 방향 (Direction / Sign) 확인")
        dir_choice = input("\n모든 관절의 방향 부호를 기본값(+1)으로 설정할까요? (Y/n): ").strip().lower()
        if dir_choice in ["", "y", "yes"]:
            for i in range(1, 7):
                calib_data["joints"][f"joint_{i}"]["direction"] = 1
        else:
            for i in range(1, 7):
                val = input(f" - {JOINT_NAMES[i-1]} 방향 (+1 / -1) [기본 +1]: ").strip()
                calib_data["joints"][f"joint_{i}"]["direction"] = -1 if val == "-1" else 1

        # Step 3
        print_header("Step 3: Follower 팔 안전 가동 범위 설정")
        import select
        min_pos = {i: 4095 for i in range(1, 7)}
        max_pos = {i: 0 for i in range(1, 7)}

        while True:
            pos = get_driver_positions(f_driver)
            for i in range(1, 7):
                if pos[i] is not None:
                    min_pos[i] = min(min_pos[i], pos[i])
                    max_pos[i] = max(max_pos[i], pos[i])

            status_parts = []
            for i in range(1, 7):
                curr = str(pos.get(i, "Err"))
                status_parts.append(f"F_J{i}:[{min_pos[i]:>4}~{max_pos[i]:>4} (현재:{curr:>4})]")

            sys.stdout.write("\r" + " | ".join(status_parts))
            sys.stdout.flush()

            if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.readline()
                print("\n\n✅ Follower 가동 범위 측정 완료!")
                break

        for i in range(1, 7):
            f_min = max(0, min_pos[i] - 15)
            f_max = min(4095, max_pos[i] + 15)
            calib_data["joints"][f"joint_{i}"]["follower_min"] = f_min
            calib_data["joints"][f"joint_{i}"]["follower_max"] = f_max
            calib_data["joints"][f"joint_{i}"]["min"] = f_min
            calib_data["joints"][f"joint_{i}"]["max"] = f_max

        # Step 4
        print_header("Step 4: 그리퍼 (Gripper) 열림 / 닫힘 설정")
        l_open, f_open = live_monitor_two_arms(l_driver, f_driver, prompt="👉 그리퍼를 '완전히 연 상태'에서 [Enter]를 누르세요.")
        l_close, f_close = live_monitor_two_arms(l_driver, f_driver, prompt="👉 그리퍼를 '완전히 닫은 상태'에서 [Enter]를 누르세요.")

        calib_data["gripper"] = {
            "leader_open": l_open[6],
            "leader_close": l_close[6],
            "follower_open": f_open[6],
            "follower_close": f_close[6],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=4, ensure_ascii=False)

        print_header("🎉 캘리브레이션 완료!")
        print(f"설정 파일 저장: {os.path.abspath(save_path)}")

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
    parser.add_argument(
        "--follower-only",
        action="store_true",
        help="팔로워 팔만 단독 캘리브레이션 (리더 팔 연결 불필요, 기존 리더 설정 보존)",
    )
    args = parser.parse_args()

    if args.follower_only:
        calibrate_follower_only(
            follower_port=args.follower,
            save_path=args.output,
        )
    else:
        calibrate(
            leader_port=args.leader,
            follower_port=args.follower,
            save_path=args.output,
        )


if __name__ == "__main__":
    main()
