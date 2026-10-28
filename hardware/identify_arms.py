"""SO-101 리더/팔로워 로봇 자동 식별 및 고유 시리얼 ID 영구 등록 도구.

15초 동안 실시간으로 움직임을 모니터링하여, 사용자가 리더암을 흔드는 순간
즉각 감지하고 고유 시리얼 ID를 영구 등록합니다.
"""

import glob
import json
import os
import subprocess
import sys
import time

try:
    from hardware.sts3215_driver import STS3215Driver
except ImportError:
    from sts3215_driver import STS3215Driver


def get_serial_number(device_path: str) -> str:
    """udevadm을 통해 tty 장치의 고유 USB 시리얼 번호 추출"""
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "-q", "property", "-n", device_path],
            text=True
        )
        for line in out.splitlines():
            if line.startswith("ID_SERIAL_SHORT="):
                return line.split("=")[1].strip()
    except Exception:
        pass
    return ""


def find_port_by_serial(target_serial: str, default_port: str = None) -> str:
    """등록된 고유 USB 시리얼 번호로 현재 연결된 tty 포트 자동 탐색"""
    if not target_serial:
        return default_port

    available_ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    for port in available_ports:
        sn = get_serial_number(port)
        if sn and sn == target_serial:
            return port

    # 매칭 실패 시 기본 포트가 존재하면 기본 포트 반환
    if default_port and os.path.exists(default_port):
        return default_port
    # 그것도 없으면 첫 번째 가용 포트 반환 (포트가 딱 1개 연결된 경우 유용)
    if available_ports:
        return available_ports[0]
    return default_port


def main():
    print("=" * 65)
    print(" 🔍 [SO-101 리더 / 팔로워 실시간 자동 식별 도구]")
    print("=" * 65)

    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if len(ports) < 2:
        print(f"❌ 오류: 로봇 시리얼 포트가 2개 이상 필요합니다. 감지된 포트: {ports}")
        sys.exit(1)

    port_serials = {}
    drivers = {}

    print("\n[1단계] 장치 초기화 및 토크 안전 해제 중...")
    for p in ports[:2]:
        sn = get_serial_number(p)
        port_serials[p] = sn
        print(f" • 포트: {p} ➔ USB 고유 시리얼: {sn if sn else '알 수 없음'}")
        d = STS3215Driver(p, baudrate=1000000, timeout=0.03)
        if not d.connect():
            print(f"❌ {p} 연결 실패!")
            sys.exit(1)
        # 안전을 위해 양쪽 모두 토크 완전 해제
        d.set_all_torques(range(1, 7), False)
        drivers[p] = d

    print("\n✅ 양쪽 로봇의 토크를 모두 해제했습니다. (안전 상태)")
    print("-" * 65)
    print(" 📢 [준비] 지금부터 👉 '리더암(조종기)' 👈 만 손으로 흔들어주세요!")
    print("    (최대 15초간 대기하며 움직임이 감지되면 즉시 리더로 확정합니다)")
    print("-" * 65)

    p_a, p_b = list(drivers.keys())[:2]
    start_pos_a = drivers[p_a].read_positions(range(1, 7))
    start_pos_b = drivers[p_b].read_positions(range(1, 7))

    max_delta_a = 0
    max_delta_b = 0
    detected_leader = None

    timeout_s = 15.0
    t_start = time.time()

    while time.time() - t_start < timeout_s:
        curr_a = drivers[p_a].read_positions(range(1, 7))
        curr_b = drivers[p_b].read_positions(range(1, 7))

        for i in range(1, 7):
            if start_pos_a.get(i) is not None and curr_a.get(i) is not None:
                max_delta_a = max(max_delta_a, abs(curr_a[i] - start_pos_a[i]))
            if start_pos_b.get(i) is not None and curr_b.get(i) is not None:
                max_delta_b = max(max_delta_b, abs(curr_b[i] - start_pos_b[i]))

        remaining = int(timeout_s - (time.time() - t_start))
        sys.stdout.write(
            f"\r ⏳ [감지 대기 {remaining:2d}s] {p_a} 움직임: {max_delta_a:4d} raw | {p_b} 움직임: {max_delta_b:4d} raw"
        )
        sys.stdout.flush()

        # 충분한 움직임 차이가 감지되면 즉시 확정 (임계값 120 raw)
        if max_delta_a > 120 and max_delta_a > max_delta_b + 80:
            detected_leader = p_a
            break
        elif max_delta_b > 120 and max_delta_b > max_delta_a + 80:
            detected_leader = p_b
            break

        time.sleep(0.1)

    print()
    for p in drivers:
        drivers[p].close()

    if not detected_leader:
        print("\n⚠️ 15초 동안 리더암의 충분한 움직임이 감지되지 않았습니다.")
        print("다시 실행하시거나 터미널에서 'python3 hardware/identify_arms.py'를 직접 실행해 보세요.")
        sys.exit(1)

    leader_port = detected_leader
    follower_port = p_b if leader_port == p_a else p_a
    leader_serial = port_serials[leader_port]
    follower_serial = port_serials[follower_port]

    print("\n" + "=" * 65)
    print(" 🎉 [식별 성공! 리더암 움직임 감지 완료]")
    print(f" 🎮 리더암(조종기) : {leader_port} (고유 시리얼: {leader_serial})")
    print(f" 🦾 팔로워암(로봇) : {follower_port} (고유 시리얼: {follower_serial})")
    print("=" * 65)

    # 설정 파일 업데이트
    cfg_paths = ["configs/calibration.json", "calibration.json"]
    for path in cfg_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["leader_port"] = leader_port
            cfg["follower_port"] = follower_port
            cfg["leader_serial"] = leader_serial
            cfg["follower_serial"] = follower_serial
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
            print(f"✅ '{path}' 설정 업데이트 완료!")


if __name__ == "__main__":
    main()
