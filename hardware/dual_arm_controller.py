"""SO-101 양팔(Dual SO-101, 총 12 DOF) 로봇 제어기.

왼팔 6 DOF (관절 1~5 + 그리퍼 6) + 오른팔 6 DOF (관절 1~5 + 그리퍼 6)
리더 암(Leader) 및 팔로워 암(Follower) 시리얼 동기화 제어.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .sts3215_driver import STS3215Driver


def raw_to_rad(raw_val: Optional[int]) -> float:
    """STS3215 Raw (0~4095) -> 라디안 (-pi ~ +pi)"""
    if raw_val is None:
        raw_val = 2048
    return float((raw_val - 2048) * (2.0 * np.pi / 4096.0))


def rad_to_raw(rad_val: float) -> int:
    """라디안 (-pi ~ +pi) -> STS3215 Raw (0~4095)"""
    raw = int(round(rad_val * (4096.0 / (2.0 * np.pi)) + 2048))
    return max(0, min(4095, raw))


class DualArmController:
    """양팔 SO-101 (총 12개 모터) 실시간 제어기"""

    def __init__(
        self,
        left_follower_port: str = "/dev/ttyACM0",
        right_follower_port: str = "/dev/ttyACM1",
        left_leader_port: Optional[str] = "/dev/ttyACM2",
        right_leader_port: Optional[str] = "/dev/ttyACM3",
        baudrate: int = 1000000,
        joint_ids: Optional[List[int]] = None,
    ):
        self.joint_ids = joint_ids if joint_ids is not None else [1, 2, 3, 4, 5, 6]

        # 팔로워 드라이버 (실제 작업 수행 암)
        self.l_follower = STS3215Driver(left_follower_port, baudrate=baudrate, timeout=0.03)
        self.r_follower = STS3215Driver(right_follower_port, baudrate=baudrate, timeout=0.03)

        # 리더 드라이버 (교시 조작용 암)
        self.l_leader = (
            STS3215Driver(left_leader_port, baudrate=baudrate, timeout=0.03) if left_leader_port else None
        )
        self.r_leader = (
            STS3215Driver(right_leader_port, baudrate=baudrate, timeout=0.03) if right_leader_port else None
        )

        self.is_connected = False

    def connect(self) -> bool:
        """모든 연결된 포트 초기화"""
        lf_ok = self.l_follower.connect()
        rf_ok = self.r_follower.connect()

        ll_ok = self.l_leader.connect() if self.l_leader else True
        rl_ok = self.r_leader.connect() if self.r_leader else True

        if not (lf_ok and rf_ok):
            print(f"⚠️ [DualArm] 팔로워 로봇 암 연결 실패: Left={lf_ok}, Right={rf_ok}")
            self.is_connected = False
            return False

        # 리더 암 토크 해제 (수동 조작 가능하도록)
        if self.l_leader and ll_ok:
            for j_id in self.joint_ids:
                self.l_leader.set_torque_enable(j_id, False)
        if self.r_leader and rl_ok:
            for j_id in self.joint_ids:
                self.r_leader.set_torque_enable(j_id, False)

        # 팔로워 암 토크 인가
        for j_id in self.joint_ids:
            self.l_follower.set_torque_enable(j_id, True)
            self.r_follower.set_torque_enable(j_id, True)

        self.is_connected = True
        print(f"✅ [DualArm] 양팔 팔로워 로봇 암 연결 성공! (총 12 DOF)")
        return True

    def read_follower_qpos(self) -> np.ndarray:
        """팔로워 암 현재 12개 관절 각도(라디안) 읽기: [Left 6개, Right 6개]"""
        if not self.is_connected:
            return np.zeros(12, dtype=np.float32)

        qpos = []
        for j_id in self.joint_ids:
            raw = self.l_follower.get_present_position(j_id)
            qpos.append(raw_to_rad(raw))

        for j_id in self.joint_ids:
            raw = self.r_follower.get_present_position(j_id)
            qpos.append(raw_to_rad(raw))

        return np.array(qpos, dtype=np.float32)

    def read_leader_qpos(self) -> np.ndarray:
        """리더 암 현재 12개 관절 각도(라디안) 읽기"""
        if not self.is_connected or not self.l_leader or not self.r_leader:
            return self.read_follower_qpos()

        qpos = []
        for j_id in self.joint_ids:
            raw = self.l_leader.get_present_position(j_id)
            qpos.append(raw_to_rad(raw))

        for j_id in self.joint_ids:
            raw = self.r_leader.get_present_position(j_id)
            qpos.append(raw_to_rad(raw))

        return np.array(qpos, dtype=np.float32)

    def send_action(self, action_12d: np.ndarray):
        """12차원 목표 각도(라디안)를 양팔 팔로워에 동기 전송"""
        if not self.is_connected:
            return

        left_targets = {j_id: rad_to_raw(action_12d[i]) for i, j_id in enumerate(self.joint_ids)}
        right_targets = {j_id: rad_to_raw(action_12d[i + 6]) for i, j_id in enumerate(self.joint_ids)}

        self.l_follower.sync_write_position(left_targets)
        self.r_follower.sync_write_position(right_targets)

    def teleop_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """텔레옵 1스텝: 리더 암 각도를 읽어 팔로워 암에 전달하고 qpos와 action 반환"""
        leader_qpos = self.read_leader_qpos()
        self.send_action(leader_qpos)
        follower_qpos = self.read_follower_qpos()
        return follower_qpos, leader_qpos

    def disconnect(self):
        """종료 시 안전하게 토크 끄고 포트 해제"""
        if self.is_connected:
            try:
                for j_id in self.joint_ids:
                    self.l_follower.set_torque_enable(j_id, False)
                    self.r_follower.set_torque_enable(j_id, False)
            except Exception:
                pass
            self.l_follower.close()
            self.r_follower.close()
            if self.l_leader:
                self.l_leader.close()
            if self.r_leader:
                self.r_leader.close()
            self.is_connected = False
            print("🛑 [DualArm] 양팔 연결 해제 완료.")
