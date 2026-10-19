"""Feetech STS3215 스마트 시리얼 서보 모터 경량 통신 드라이버.

SO-101(SO-ARM100) 로봇팔 제어를 위해 pyserial 기반으로 패킷 송수신을 직접 처리합니다.
"""

import time
import serial
from typing import Dict, List, Optional, Tuple, Union


class STS3215Driver:
    # Instruction 코딩
    INST_PING = 0x01
    INST_READ = 0x02
    INST_WRITE = 0x03
    INST_REG_WRITE = 0x04
    INST_ACTION = 0x05
    INST_SYNC_WRITE = 0x83

    # 레지스터 주소
    ADDR_ID = 5
    ADDR_BAUDRATE = 6
    ADDR_TORQUE_ENABLE = 40
    ADDR_ACC = 41
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_LOCK = 55
    ADDR_PRESENT_POSITION = 56
    ADDR_PRESENT_SPEED = 58
    ADDR_PRESENT_LOAD = 60
    ADDR_PRESENT_VOLTAGE = 62
    ADDR_PRESENT_TEMPERATURE = 63
    ADDR_MOVING = 66

    # STS3215 특성
    RESOLUTION = 4096  # 0 ~ 4095 (12-bit)
    CENTER_POSITION = 2048

    def __init__(self, port: str, baudrate: int = 1000000, timeout: float = 0.05):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None

    def connect(self) -> bool:
        """시리얼 포트 연결"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self.ser.flushInput()
            self.ser.flushOutput()
            return True
        except serial.SerialException as err:
            print(f"[STS3215] 포트 열기 실패 ({self.port}): {err}")
            self.ser = None
            return False

    def close(self):
        """시리얼 포트 닫기"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _calc_checksum(servo_id: int, length: int, instruction: int, params: List[int]) -> int:
        """체크섬 계산 (~(ID + Length + Instruction + sum(Params)) & 0xFF)"""
        val = servo_id + length + instruction + sum(params)
        return (~val) & 0xFF

    def _send_packet(self, servo_id: int, instruction: int, params: List[int]) -> bool:
        """패킷 구성 후 시리얼 전송"""
        if not self.ser or not self.ser.is_open:
            return False
        length = len(params) + 2
        checksum = self._calc_checksum(servo_id, length, instruction, params)
        packet = bytearray([0xFF, 0xFF, servo_id, length, instruction] + params + [checksum])
        self.ser.write(packet)
        return True

    def _read_response(self, expected_id: int, expected_data_len: int = 0) -> Optional[List[int]]:
        """서보 모터 응답 수신 파싱"""
        if not self.ser or not self.ser.is_open:
            return None

        # 헤더 0xFF, 0xFF 탐색
        start_time = time.time()
        head_match = 0
        while head_match < 2:
            if time.time() - start_time > self.timeout:
                return None
            b = self.ser.read(1)
            if not b:
                return None
            if b[0] == 0xFF:
                head_match += 1
            else:
                head_match = 0

        # 헤더 이후 기본 정보 읽기: ID(1), Length(1), Error(1)
        header_rem = self.ser.read(3)
        if len(header_rem) < 3:
            return None
        res_id, length, err = header_rem[0], header_rem[1], header_rem[2]

        if expected_id != 0xFE and res_id != expected_id:
            return None

        # 데이터(length - 2 바이트) 및 체크섬(1 바이트) 읽기
        data_len = length - 2
        data_and_check = self.ser.read(data_len + 1)
        if len(data_and_check) < data_len + 1:
            return None

        params = list(data_and_check[:data_len])
        checksum = data_and_check[-1]

        # 체크섬 검증
        calc_check = self._calc_checksum(res_id, length, err, params)
        if calc_check != checksum:
            return None

        return params

    def ping(self, servo_id: int) -> bool:
        """모터 연결 여부 Ping 확인"""
        if not self.ser or not self.ser.is_open:
            return False
        self.ser.flushInput()
        self._send_packet(servo_id, self.INST_PING, [])
        res = self._read_response(servo_id)
        return res is not None

    def set_torque(self, servo_id: int, enable: bool) -> bool:
        """서보 모터 토크 ON/OFF 설정 (enable=False 시 수동 조작 가능)"""
        val = 1 if enable else 0
        if not self.ser or not self.ser.is_open:
            return False
        self.ser.flushInput()
        ok = self._send_packet(servo_id, self.INST_WRITE, [self.ADDR_TORQUE_ENABLE, val])
        self.ser.flush()
        return ok

    def set_all_torques(self, servo_ids: List[int], enable: bool):
        """여러 서보 모터의 토크 일괄 설정 (브로드캐스트 + 개별 확인사살)"""
        if not self.ser or not self.ser.is_open:
            return
        val = 1 if enable else 0
        
        # 1. 브로드캐스트 ID(0xFE)로 버스 전체에 한 번에 명령 전송
        self.ser.flushInput()
        self._send_packet(0xFE, self.INST_WRITE, [self.ADDR_TORQUE_ENABLE, val])
        self.ser.flush()
        time.sleep(0.01)

        # 2. 각 모터별로도 개별 전송하여 누락 방지 (패킷 간 5ms 딜레이)
        for sid in servo_ids:
            self.set_torque(sid, enable)
            time.sleep(0.005)

        # 3. 토크 해제(False)일 경우, 완전히 풀리도록 브로드캐스트 한 번 더 재전송
        if not enable:
            time.sleep(0.01)
            self._send_packet(0xFE, self.INST_WRITE, [self.ADDR_TORQUE_ENABLE, 0])
            self.ser.flush()
            time.sleep(0.02)

    def read_position(self, servo_id: int) -> Optional[int]:
        """현재 모터 위치 (0 ~ 4095) 읽기"""
        if not self.ser or not self.ser.is_open:
            return None
        self.ser.flushInput()
        # ADDR_PRESENT_POSITION 주소에서 2바이트 읽기
        self._send_packet(servo_id, self.INST_READ, [self.ADDR_PRESENT_POSITION, 2])
        res = self._read_response(servo_id, expected_data_len=2)
        if res and len(res) == 2:
            return res[0] | (res[1] << 8)  # Little-endian
        return None

    def read_positions(self, servo_ids: List[int]) -> Dict[int, Optional[int]]:
        """여러 모터의 현재 위치 순차 읽기"""
        positions = {}
        for sid in servo_ids:
            positions[sid] = self.read_position(sid)
        return positions

    def write_position(self, servo_id: int, position: int, speed: int = 0, acc: int = 0) -> bool:
        """단일 모터 목표 위치 지령 (position: 0~4095, speed: 0~4095, acc: 0~254)"""
        position = max(0, min(4095, int(position)))
        pos_l = position & 0xFF
        pos_h = (position >> 8) & 0xFF
        time_l = 0
        time_h = 0
        spd_l = speed & 0xFF
        spd_h = (speed >> 8) & 0xFF

        # [ADDR_ACC, acc, pos_l, pos_h, time_l, time_h, spd_l, spd_h]
        params = [self.ADDR_ACC, acc, pos_l, pos_h, time_l, time_h, spd_l, spd_h]
        return self._send_packet(servo_id, self.INST_WRITE, params)

    def sync_write_positions(
        self,
        id_pos_dict: Dict[int, int],
        speed: int = 0,
        acc: int = 0,
    ) -> bool:
        """여러 모터에 목표 위치를 동시(Sync Write) 지령.
        id_pos_dict: {servo_id: goal_position}
        """
        if not self.ser or not self.ser.is_open or not id_pos_dict:
            return False

        # 레지스터: ADDR_ACC부터 총 7바이트 (acc 1B + pos 2B + time 2B + speed 2B)
        data_len_per_servo = 7
        sync_params = [self.ADDR_ACC, data_len_per_servo]

        for sid, pos in id_pos_dict.items():
            pos = max(0, min(4095, int(pos)))
            sync_params.extend([
                sid,
                acc,
                pos & 0xFF,
                (pos >> 8) & 0xFF,
                0,
                0,  # time
                speed & 0xFF,
                (speed >> 8) & 0xFF,  # speed
            ])

        # Sync write는 브로드캐스트 ID(0xFE)로 전송
        return self._send_packet(0xFE, self.INST_SYNC_WRITE, sync_params)

    def read_status(self, servo_id: int) -> Optional[Dict[str, Union[int, float]]]:
        """모터 상태 상세 진단 (위치, 전압, 온도, 부하)"""
        if not self.ser or not self.ser.is_open:
            return None
        self.ser.flushInput()
        # ADDR_PRESENT_POSITION(56)부터 8바이트 (pos 2B, spd 2B, load 2B, volt 1B, temp 1B)
        self._send_packet(servo_id, self.INST_READ, [self.ADDR_PRESENT_POSITION, 8])
        res = self._read_response(servo_id, expected_data_len=8)
        if res and len(res) >= 8:
            pos = res[0] | (res[1] << 8)
            spd = res[2] | (res[3] << 8)
            load = res[4] | (res[5] << 8)
            volt = res[6] / 10.0  # 0.1V 단위
            temp = res[7]  # 섭씨 도
            return {
                "id": servo_id,
                "position": pos,
                "speed": spd,
                "load": load,
                "voltage": volt,
                "temperature": temp,
            }
        return None
