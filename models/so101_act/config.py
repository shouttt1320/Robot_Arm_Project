"""SO-101 픽앤플레이스 ACT 모델 및 학습 설정 (Config)."""

import os

def _find_project_root():
    cur = os.path.abspath(__file__)
    for _ in range(5):
        cur = os.path.dirname(cur)
        if os.path.exists(os.path.join(cur, "configs", "calibration.json")) or os.path.exists(os.path.join(cur, "hardware")):
            return cur
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = _find_project_root()
DATA_DIR = os.path.join(BASE_DIR, "data", "so101_pick_place")
CKPT_DIR = os.path.join(BASE_DIR, "ckpt", "so101_act")

# 2. 로봇 및 센서 스펙
TASK_NAME = "so101_pick_place"
STATE_DIM = 6  # 6-DOF (1: Base, 2: Shoulder, 3: Elbow, 4: Wrist Pitch, 5: Wrist Roll, 6: Gripper)
CAMERA_NAMES = ["top", "wrist"]  # Top 뷰 + Wrist 뷰 듀얼 카메라
CONTROL_FPS = 30  # 30Hz

# 3. ACT 모델 하이퍼파라미터
CHUNK_SIZE = 50           # 미래 액션 예측 청크 크기 (num_queries = 50, 약 1.67초 분량)
HIDDEN_DIM = 512          # 트랜스포머 임베딩 차원
DIM_FEEDFORWARD = 3200    # 피드포워드 레이어 차원
ENC_LAYERS = 6            # 인코더 레이어 수
DEC_LAYERS = 7            # 디코더 레이어 수
NHEADS = 8                # 어텐션 헤드 수
BACKBONE = "resnet18"     # 이미지 백본
LATENT_DIM = 32           # CVAE 잠재 변수 z 차원
KL_WEIGHT = 0.1           # CVAE KL 발산 손실 가중치 (0.1이 공 피처맵 보존에 최적)
LR = 1e-5                  # 트랜스포머 학습률
LR_BACKBONE = 1e-5         # ResNet 백본 학습률
WEIGHT_DECAY = 1e-4

# 4. 학습 설정
BATCH_SIZE = 8
NUM_EPOCHS = 5000          # 5,000 에포크 기본 설정 (수천 에포크에서 시각 어텐션 활성화)
SAVE_EVERY = 500           # 체크포인트 저장 주기 (500 에포크 단위)
SEED = 0

# 5. 실 로봇 추론(Deployment) 파라미터
TEMPORAL_AGG_M = 0.02     # Temporal Ensembling 지수 가중치 m (exp(-m * i))
