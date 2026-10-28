"""SO-101 양팔(Dual SO-101, 12 DOF) 로봇 암 사전 점검 및 텔레옵 추종 테스트 도구.

기능:
1. 12개 서보 모터(STS3215) 통신, 전압, 온도, 초기 각도 점검
2. 실시간 텔레옵 추종 테스트 (Leader -> Follower) 및 관절별 추종 오차(Tracking Error) 모니터링
3. 리더 암이 없을 경우를 위한 팔로워 암 안전 동작 자가 테스트 (Wave Test)
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# 프로젝트 루트 임포트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.sts3215_driver import STS3215Driver
from hardware.dual_arm_controller import DualArmController, raw_to_rad, rad_to_raw


def ping_and_inspect_arm(driver: STS3215Driver, arm_name: str, joint_ids=[1, 2, 3, 4, 5, 6]) -> bool:
    """단일 로봇 팔의 6개 모터 통신 및 상태 테이블 출력"""
    print(f"\n🔍 [{arm_name} 포트 점검: {driver.port}]")
    if not driver.connect():
        print(f"❌ 포트 연결 실패: {driver.port}")
        return False

    print(f"┌───────┬────────────┬─────────────┬─────────────┬───────────┬───────────┐")
    print(f"│ 모터  │ 연결 상태  │ Raw Pos     │ 각도 (deg)  │ 전압 (V)  │ 온도 (°C) │")
    print(f"├───────┼────────────┼─────────────┼─────────────┼───────────┼───────────┤")

    all_ok = True
    for j_id in joint_ids:
        pos = driver.get_present_position(j_id)
        if pos is not None:
            deg = (pos - 2048) * (360.0 / 4096.0)
            volt = driver.get_present_voltage(j_id)
            volt_str = f"{volt:.1f}V" if volt is not None else "N/A"
            temp = driver.get_present_temperature(j_id)
            temp_str = f"{temp}°C" if temp is not None else "N/A"
            print(f"│ ID {j_id:02d} │    정상    │ {pos:4d} / 4095 │ {deg:+6.1f}°     │ {volt_str:>9} │ {temp_str:>9} │")
        else:
            print(f"│ ID {j_id:02d} │ ❌ 무응답  │     ----    │    ----     │      ---- │      ---- │")
            all_ok = False

    print(f"└───────┴────────────┴─────────────┴─────────────┴───────────┴───────────┘")
    return all_ok


def run_teleop_tracking_test(controller: DualArmController, fps: int = 20):
    """실시간 양팔 텔레옵 추종 및 오차 모니터링"""
    print("\n" + "=" * 70)
    print(" 🎮 [양팔 실시간 텔레옵 추종 테스트]")
    print(" • 리더 암을 손으로 움직이면 팔로워 암이 실시간으로 따라 움직입니다.")
    print(" • 콘솔에 12개 관절의 추종 오차(Error = |Leader - Follower|)가 실시간 표시됩니다.")
    print(" • [Ctrl + C]를 누르면 안전하게 종료됩니다.")
    print("=" * 70 + "\n")

    dt = 1.0 / fps

    try:
        step = 0
        while True:
            t0 = time.time()
            follower_qpos, leader_qpos = controller.teleop_step()

            error_rad = np.abs(leader_qpos - follower_qpos)
            error_deg = np.rad2deg(error_rad)

            max_err_l = np.max(error_deg[:6])
            max_err_r = np.max(error_deg[6:])

            # 1줄 실시간 상태 갱신
            left_str = " ".join([f"{e:4.1f}" for e in error_deg[:6]])
            right_str = " ".join([f"{e:4.1f}" for e in error_deg[6:]])

            sys.stdout.write(
                f"\r[Step {step:04d}] Left Arm Err(deg): [{left_str}] (Max: {max_err_l:4.1f}°) | "
                f"Right Arm Err(deg): [{right_str}] (Max: {max_err_r:4.1f}°)"
            )
            sys.stdout.flush()

            step += 1
            elapsed = time.time() - t0
            sleep_t = max(0.0, dt - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n\n🛑 텔레옵 테스트를 종료합니다.")


def run_safe_wave_test(controller: DualArmController):
    """리더 암 없이 팔로워 암의 관절을 작은 각도(±5도)로 부드럽게 움직여보는 자가 진단"""
    print("\n" + "=" * 70)
    print(" 🌊 [팔로워 양팔 부드러운 자가 진단 (Safe Wave Test)]")
    print(" • 현재 위치를 기준으로 작은 각도(±5도)를 왕복하며 12개 모터의 정상 구동을 검증합니다.")
    print(" • [Ctrl + C]를 누르면 즉시 중지됩니다.")
    print("=" * 70 + "\n")

    current_qpos = controller.read_follower_qpos()
    print("📍 기준 위치 기록 완료. 테스트 시작...")

    t = 0.0
    dt = 0.05
    amplitude = np.deg2rad(6.0)  # 안전한 6도 진폭

    try:
        while t < 10.0:  # 10초간 테스트
            delta = amplitude * np.sin(2.0 * np.pi * 0.5 * t)
            target = current_qpos.copy()
            # 1번(Base), 4번(Wrist), 6번(Gripper) 부드럽게 사인파 구동
            target[0] += delta
            target[4] += delta
            target[6] += delta
            target[10] += delta

            controller.send_action(target)
            actual = controller.read_follower_qpos()

            sys.stdout.write(
                f"\r⏱️ 진행 시간: {t:4.1f}s / 10.0s | Target Delta: {np.rad2deg(delta):+4.1f}° | "
                f"L-Base: {np.rad2deg(actual[0]):+5.1f}°, R-Base: {np.rad2deg(actual[6]):+5.1f}°"
            )
            sys.stdout.flush()

            t += dt
            time.sleep(dt)

        # 원래 위치로 복귀
        print("\n\n📍 원래 위치로 복귀 중...")
        controller.send_action(current_qpos)
        time.sleep(1.0)
        print("✅ 자가 진단 완료! 모든 모터가 원활하게 응답합니다.")

    except KeyboardInterrupt:
        print("\n\n🛑 사용자에 의해 자가 진단이 중단되었습니다. 원래 위치로 복귀합니다.")
        controller.send_action(current_qpos)


def main():
    parser = argparse.ArgumentParser(description="SO-101 양팔 사전 점검 및 텔레옵 테스트")
    parser.add_argument("--left_follower", type=str, default="/dev/ttyACM0", help="Left Follower 포트")
    parser.add_argument("--right_follower", type=str, default="/dev/ttyACM1", help="Right Follower 포트")
    parser.add_argument("--left_leader", type=str, default="/dev/ttyACM2", help="Left Leader 포트 (생략 가능)")
    parser.add_argument("--right_leader", type=str, default="/dev/ttyACM3", help="Right Leader 포트 (생략 가능)")
    parser.add_argument("--wave_test", action="store_true", help="리더 암 없이 팔로워 암 자가 진단 실행")
    args = parser.parse_args()

    print("=" * 70)
    print(" 🦾 [SO-101 양팔 (12 DOF) 하드웨어 사전 점검기]")
    print(f" • Left Follower  : {args.left_follower}")
    print(f" • Right Follower : {args.right_follower}")
    print(f" • Left Leader    : {args.left_leader}")
    print(f" • Right Leader   : {args.right_leader}")
    print("=" * 70)

    # 1. 각 포트별 모터 핑 및 진단
    controller = DualArmController(
        left_follower_port=args.left_follower,
        right_follower_port=args.right_follower,
        left_leader_port=args.left_leader,
        right_leader_port=args.right_leader,
    )

    lf_ok = ping_and_inspect_arm(controller.l_follower, "Left Follower (좌측 작업 팔)")
    rf_ok = ping_and_inspect_arm(controller.r_follower, "Right Follower (우측 작업 팔)")

    if not (lf_ok and rf_ok):
        print("\n❌ 팔로워 암의 일부 모터가 응답하지 않습니다. 케이블 연결과 전원을 확인해 주세요.")
        controller.disconnect()
        return

    controller.connect()

    # 2. 동작 테스트 수행
    if args.wave_test:
        run_safe_wave_test(controller)
    else:
        if controller.l_leader and controller.r_leader:
            ping_and_inspect_arm(controller.l_leader, "Left Leader (좌측 조작 팔)")
            ping_and_inspect_arm(controller.r_leader, "Right Leader (우측 조작 팔)")
            run_teleop_tracking_test(controller)
        else:
            print("\nℹ️ 리더 암 포트가 연결되지 않아 자가 진단(Wave Test)을 실행합니다.")
            run_safe_wave_test(controller)

    controller.disconnect()


if __name__ == "__main__":
    main()
