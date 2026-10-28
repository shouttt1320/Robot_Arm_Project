"""양안(Dual Eyes) 카메라 및 DINOv2 기반 ACT 모델 및 학습 설정 (Config)."""

import os

def _find_project_root():
    cur = os.path.abspath(__file__)
    for _ in range(5):
        cur = os.path.dirname(cur)
        if os.path.exists(os.path.join(cur, "configs", "calibration.json")) or os.path.exists(os.path.join(cur, "hardware")):
            return cur
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = _find_project_root()
DATA_DIR = os.path.join(BASE_DIR, "data", "so101_dual_eyes")
CKPT_DIR = os.path.join(BASE_DIR, "ckpt", "so101_dual_dinov2_act")

# 1. 로봇 및 센서 스펙
TASK_NAME = "so101_dual_eyes_pick_place"
STATE_DIM = 6  # 6-DOF (1: Base, 2: Shoulder, 3: Elbow, 4: Wrist Pitch, 5: Wrist Roll, 6: Gripper)
CAMERA_NAMES = ["cam_left", "cam_right"]  # 좌안 / 우안 스테레오 듀얼 카메라
CAMERA_PORTS = {"cam_left": 4, "cam_right": 6}  # 기본 하드웨어 포트 (video4: 좌안, video6: 우안)
DISABLE_AUTOFOCUS = True  # 학습 데이터 일관성을 위한 오토포커스 끄기 및 고정
FIXED_FOCUS_VALUE = 25    # 작업대 픽앤플레이스 거리에 최적화된 고정 초점 값 (0~255)
CONTROL_FPS = 30  # 30Hz

# 2. 이미지 전처리 및 해상도 (DINOv2 패치 크기 14의 배수)
# 480x640(3:4 비율)에 최적화된 14배수: 224(=14x16) x 294(=14x21), 왜곡 없는 비율 보존
IMAGE_SIZE = (224, 294)  # (Height, Width)

# 3. DINOv2 백본 및 ACT 모델 하이퍼파라미터
BACKBONE = "dinov2_vits14"     # "dinov2_vits14", "dinov2_vitb14"
TRAIN_BACKBONE = False         # DINOv2 사전학습 가중치 고정(Freeze) 여부
CHUNK_SIZE = 50                # 미래 액션 예측 청크 크기 (약 1.67초 분량)
HIDDEN_DIM = 512               # 트랜스포머 임베딩 차원
DIM_FEEDFORWARD = 3200         # 피드포워드 레이어 차원
ENC_LAYERS = 6                 # 인코더 레이어 수
DEC_LAYERS = 7                 # 디코더 레이어 수
NHEADS = 8                     # 어텐션 헤드 수
LATENT_DIM = 32                # CVAE 잠재 변수 z 차원
KL_WEIGHT = 0.1                # CVAE KL 발산 손실 가중치

# 4. 학습 최적화 설정
LR = 1e-5                      # 트랜스포머 및 프로젝션 학습률
LR_BACKBONE = 1e-5             # 백본 학습률 (TRAIN_BACKBONE=True 시 적용)
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 3000
SAVE_EVERY = 200
SEED = 0

# 5. 실 로봇 추론(Deployment) 파라미터
TEMPORAL_AGG_M = 0.02          # Temporal Ensembling 지수 가중치 m (exp(-m * i))
