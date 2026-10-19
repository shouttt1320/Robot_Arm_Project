# 🦾 SO-101 (SO-ARM100) ACT 모방 학습 전체 운영 가이드

SO-101 6-DOF 로봇 팔과 듀얼 RGB 카메라(Top, Wrist)를 이용한 **데이터 수집 ➔ 검증 ➔ ACT 정책 학습 ➔ 실시간 자율 추론(배포)** 전체 파이프라인 가이드입니다.

---

## 📌 목차
1. [시스템 환경 및 사전 준비](#1-시스템-환경-및-사전-준비)
2. [전체 파이프라인 흐름도](#2-전체-파이프라인-흐름도)
3. [Step 1: 하드웨어 및 센서 연결 확인](#step-1-하드웨어-및-센서-연결-확인)
4. [Step 2: 텔레옵 및 에피소드 데이터 수집](#step-2-텔레옵-및-에피소드-데이터-수집)
5. [Step 3: 데이터셋 품질 검증 및 시각화](#step-3-데이터셋-품질-검증-및-시각화)
6. [Step 4: ACT 모델 학습 (GPU 가속)](#step-4-act-모델-학습-gpu-가속)
7. [Step 5: 실시간 자율 구동 (배포 & 안전 홈 복귀)](#step-5-실시간-자율-구동-배포--안전-홈-복귀)
8. [자주 묻는 질문 및 트러블슈팅 (FAQ)](#자주-묻는-질문-및-트러블슈팅-faq)

---

## 1. 시스템 환경 및 사전 준비

* **OS**: Ubuntu 24.04 LTS (x86_64)
* **Python**: 3.12 (가상환경)
* **핵심 라이브러리**:
  * PyTorch with CUDA 가속 (`torch 2.14.1+cu130`)
  * NumPy 1.x (`numpy==1.26.4` - OpenCV 바이너리 호환 필수)
  * OpenCV (`opencv-python 4.6.0+`)
* **하드웨어 인터페이스**:
  * **Follower (구동 로봇)**: `/dev/ttyACM0` (모터 ID 1~6, STS3215 서보)
  * **Leader (조작 로봇)**: `/dev/ttyACM1` (모터 ID 1~6, 무토크 원격 조작기)
  * **Top 카메라 (상단 시점)**: `/dev/video4` (640x480 @ 30fps)
  * **Wrist 카메라 (손목 시점)**: `/dev/video6` (640x480 @ 30fps)

> 💡 **시리얼 포트 권한 부여 (필요 시 최초 1회)**:
> ```bash
> sudo usermod -aG dialout $USER
> sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
> ```

---

## 2. 전체 파이프라인 흐름도

```mermaid
flowchart LR
    A[Leader 원격 조작] --> B[record_episodes.py\n30Hz HDF5 수집]
    B --> C[visualize_data.py\n무결성/영상 검증]
    C --> D[train.py\nACT CVAE + DETR 학습]
    D --> E[(policy_best.ckpt\ndataset_stats.pkl)]
    E --> F[deploy_real_robot.py\n실시간 카메라 자율 제어]
    F --> G[종료 시 안전 홈 복귀 & 토크 해제]
```

---

## 3. Step 1: 하드웨어 및 센서 연결 확인

로봇과 카메라가 정상 인식되었는지 확인합니다.

```bash
# 1. 시리얼 포트 연결 확인
ls -l /dev/ttyACM*

# 2. 카메라 디바이스 번호 확인
v4l2-ctl --list-devices

# 3. 모터 ID 및 통신 핑 스캔 (필요 시)
python3 -m hardware.scan_motors

# 4. 순수 텔레옵(리더-팔로워 동조) 단독 동작 테스트
python3 -m collection.teleop
```

---

## 4. Step 2: 텔레옵 및 에피소드 데이터 수집

원격 조작을 통해 Pick-and-Place 작업을 직접 시연하고 에피소드별 HDF5 파일로 저장합니다.

### 🚀 원클릭 실행 스크립트
```bash
./run_data_collection.sh
```
또는 직접 파이썬 실행:
```bash
python3 -m collection.record_episodes --top-cam 4 --wrist-cam 6 --fps 30
```

### 🎮 조작 인터페이스 및 기능
* **[스페이스바 (Space)]**: **녹화 시작 / 녹화 종료 (토글)**
  * 시간 제한 없이 원하는 만큼 자연스럽게 조작 가능
  * 종료 시 현재 에피소드가 `data/so101_pick_place/episode_XX.hdf5`로 즉시 저장
* **[D] 키**: 최근 녹화한 마지막 에피소드 즉시 폐기 및 재촬영
* **[Q] 키 또는 [ESC]**: 수집 프로그램 안전 종료 (팔로워 토크 자동 해제)
* **실시간 HUD**: 상단/손목 카메라 듀얼 화면, 현재 프레임 수, 녹화 시간(초), FPS 표시

---

## 5. Step 3: 데이터셋 품질 검증 및 시각화

수집된 데이터셋의 프레임 누락, 관절 한계치 초과 여부, 카메라 동기화를 검증합니다.

```bash
# 1. 전체 32개 에피소드 무결성 자동 전수 검사 (PASS/FAIL 보고서)
python3 -m collection.visualize_data --all

# 2. 특정 에피소드 듀얼 비디오 재생 및 6개 관절 궤적 그래프 동시 확인
python3 -m collection.visualize_data --episode 0

# 3. 브라우저/외부 공유용 H.264 MP4 동영상 추출
python3 -m collection.visualize_data --episode 0 --save-mp4
```

---

## 6. Step 4: ACT 모델 학습 (GPU 가속)

검증된 32개 에피소드(총 약 15,300 스텝)를 로드하여 CVAE + DETR 구조의 정책 신경망을 학습합니다.

### 🚀 원클릭 실행 스크립트
```bash
./train.sh
```
또는 직접 파이썬 옵션 지정 실행:
```bash
# 100 에포크 기본 학습
python3 -m so101_act.train --epochs 100 --batch-size 8 --lr 1e-4

# 장기 고성능 학습 (300 에포크 권장)
python3 -m so101_act.train --epochs 300 --batch-size 8 --lr 1e-4
```

### 📁 학습 결과 산출물 (`ckpt/so101_act/`)
* **`policy_best.ckpt`**: 검증 손실(Val Loss)이 가장 우수한 최적 가중치 모델
* **`dataset_stats.pkl`**: 데이터 정규화에 사용된 관절각 및 액션 평균/표준편차
* **`train_summary.csv`**: 에포크별 Train/Val Loss, L1 모방 오차, KL 발산 수치 테이블
* **`train_history.json`**: 전체 에포크 상세 로그 JSON
* **`train_curve.png`**: Total Loss, L1 Loss, KL Divergence 수렴 곡선 시각화 그래프

---

## 7. Step 5: 실시간 자율 구동 (배포 & 안전 홈 복귀)

학습된 최적 가중치(`policy_best.ckpt`)를 불러와 실제 로봇 팔이 두 대의 카메라 시각 정보만을 보고 물체를 집어 옮깁니다.

### 🚀 원클릭 실행 스크립트
```bash
./deploy.sh
```
또는 직접 파이썬 실행:
```bash
python3 -m so101_act.deploy_real_robot --max-steps 600
```

### 🌟 핵심 배포 기능
1. **Temporal Aggregation (지수 앙상블)**:
   - 예측된 청크(CHUNK_SIZE=30)의 중첩 궤적에 가중 평균을 적용하여 **떨림 없는 부드러운 움직임**을 보장합니다.
2. **자동 홈 포즈 복귀 (Safe Return-to-Home)**:
   - 추론 완료 후(또는 [Q] / [Ctrl+C] 입력 시), 현재 위치에서 시작 자세까지 **Cosine S-Curve 보간(2.5초)**으로 부드럽게 복귀합니다.
3. **안전 토크 해제**:
   - 홈 포즈에 안전하게 안착한 뒤 6개 모터의 토크가 자동으로 풀립니다.

---

## 8. 자주 묻는 질문 및 트러블슈팅 (FAQ)

### Q1. `Module compiled with NumPy 1.x cannot run in NumPy 2.x` 에러가 납니다.
* **원인**: OpenCV 4.x C-확장 모듈이 NumPy 1.x ABI로 빌드되어 발생합니다.
* **해결**: 반드시 `numpy<2` 버전을 유지해야 합니다.
  ```bash
  pip install "numpy<2"
  ```

### Q2. PyTorch가 GPU(CUDA)를 잡지 못하고 CPU로 돕니다.
* **원인**: 시스템에 CPU 전용 PyTorch가 설치되어 있는 경우입니다.
* **해결**: CUDA 지원 휠을 재설치합니다.
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```

### Q3. 시리얼 포트 접근 권한 에러 (`Permission denied: /dev/ttyACM0`)
* **해결**:
  ```bash
  sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
  ```

### Q4. 카메라가 열리지 않거나 인덱스가 바뀝니다.
* USB 허브 연결 상태에 따라 인덱스가 바뀔 수 있습니다. `v4l2-ctl --list-devices`로 확인 후 `--top-cam`, `--wrist-cam` 인자를 지정하세요.
