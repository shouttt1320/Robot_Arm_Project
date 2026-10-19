"""ACT 모델 비전 피쳐맵 및 크로스 어텐션 시각화 정밀 진단기 (개선 버전).

에피소드 데이터(또는 실시간 캡처)를 모델에 주입하여:
  1. 원본 RGB 이미지 (Top, Wrist)
  2. ResNet18 백본 피쳐맵 활성화 히트맵 (CNN 시각 특징)
  3. Transformer 디코더 Cross-Attention 히트맵 (전체 50스텝 청크 평균 또는 특정 쿼리)
  4. 🎯 정밀 좌표 변환 최고 주목점(Peak Attention Focus) 십자선 (15x20 Grid -> 640x480 Pixel)
  5. 균등 분포 대비 어텐션 밀도, 집중도(Entropy, Peak Ratio), 중앙 ROI 가중치 정량 분석
을 2행 4열의 고해상도 종합 진단 리포트 PNG로 생성합니다.
"""

import argparse
import os
import sys
import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

# 프로젝트 루트 sys.path 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from so101_act.config import (
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    HIDDEN_DIM,
    LATENT_DIM,
    NHEADS,
    STATE_DIM,
)
from so101_act.models import build_act_model


def overlay_heatmap_on_image(rgb_img: np.ndarray, heatmap_2d: np.ndarray, colormap=cv2.COLORMAP_JET, alpha=0.55):
    """2D 히트맵(15x20)을 원본 RGB 이미지(640x480)에 업샘플링하여 자연스럽게 오버레이."""
    h, w = rgb_img.shape[:2]
    # 0~1 정규화
    h_min, h_max = heatmap_2d.min(), heatmap_2d.max()
    if h_max > h_min:
        norm_map = (heatmap_2d - h_min) / (h_max - h_min)
    else:
        norm_map = np.zeros_like(heatmap_2d)

    # 원본 해상도로 부드럽게 업샘플링 (Bicubic)
    resized_map = cv2.resize(norm_map, (w, h), interpolation=cv2.INTER_CUBIC)
    uint8_map = np.uint8(255 * np.clip(resized_map, 0.0, 1.0))
    colored_map = cv2.applyColorMap(uint8_map, colormap)
    colored_rgb = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(rgb_img, 1.0 - alpha, colored_rgb, alpha, 0)
    return blended, norm_map


def inspect_vision_pipeline(
    hdf5_path: str,
    step: int = None,
    ckpt_path: str = None,
    stats_path: str = None,
    output_path: str = "vision_inspection.png",
    query: str = "all",
):
    # 1. 파일 검증
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 파일을 찾을 수 없습니다: {hdf5_path}")

    ckpt_path = ckpt_path or os.path.join(CKPT_DIR, "policy_best.ckpt")
    stats_path = stats_path or os.path.join(os.path.dirname(ckpt_path), "dataset_stats.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"가중치 파일이 없습니다: {ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"정규화 통계 파일이 없습니다: {stats_path}")

    # 2. HDF5 데이터 로드
    with h5py.File(hdf5_path, "r") as root:
        qpos_all = root["observations/qpos"][()]
        total_steps = len(qpos_all)
        if step is None:
            # 기본값: 에피소드 약 35% 시점 (공으로 접근하는 최적 프레임)
            step = int(total_steps * 0.35)
        step = max(0, min(step, total_steps - 1))

        curr_qpos = qpos_all[step]
        raw_images = {
            "top": root["observations/images/top"][step],
            "wrist": root["observations/images/wrist"][step],
        }

    # 3. 모델 빌드 및 가중치 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=CHUNK_SIZE,
        camera_names=CAMERA_NAMES,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    import pickle
    with open(stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    # 4. Attention & Feature Hooks 설치
    captured_attn = None

    def attn_hook(module, input, output):
        nonlocal captured_attn
        # PyTorch MultiheadAttention 출력: (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            captured_attn = output[1].detach().cpu()

    # 디코더의 마지막 레이어 cross-attention에 hook 장착
    last_dec_layer = model.transformer.decoder.layers[-1]
    hook_handle = last_dec_layer.multihead_attn.register_forward_hook(attn_hook)

    # 5. 전처리 및 추론
    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cam_tensors = []
    for name in ["top", "wrist"]:
        img = raw_images[name]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = normalize_img(img_t)
        cam_tensors.append(img_t)

    img_input = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(device)  # (1, 2, 3, 480, 640)

    # qpos 정규화
    qpos_norm = (curr_qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
    qpos_input = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

    # 백본 피쳐맵 직접 추출
    with torch.inference_mode():
        top_feat, _ = model.backbone(img_input[:, 0])
        wrist_feat, _ = model.backbone(img_input[:, 1])
        top_fmap = top_feat[0].squeeze(0).cpu().numpy()     # (512, 15, 20)
        wrist_fmap = wrist_feat[0].squeeze(0).cpu().numpy() # (512, 15, 20)

        # 모델 전체 순전파 (어텐션 캡처)
        a_hat, _, _ = model(qpos=qpos_input, image=img_input)

    hook_handle.remove()

    # 6. 피쳐맵 활성화 에너지(L2 Norm) 계산 (15x20)
    top_energy = np.linalg.norm(top_fmap, axis=0)       # (15, 20)
    wrist_energy = np.linalg.norm(wrist_fmap, axis=0)   # (15, 20)

    # 7. 트랜스포머 어텐션 가중치 파싱 (shape: [1, 50, 602])
    # 토큰 인덱스: [0: Latent z, 1: qpos, 2~301: Top Cam (15x20), 302~601: Wrist Cam (15x20)]
    all_queries_attn = captured_attn.squeeze(0).numpy()  # (50, 602)
    if query == "all":
        attn_weights = all_queries_attn.mean(axis=0)     # (602,)
        query_desc = "All 50 Action Queries (Mean)"
    else:
        q_idx = int(query)
        attn_weights = all_queries_attn[q_idx]          # (602,)
        query_desc = f"Action Query Slot #{q_idx}"

    z_weight = attn_weights[0]
    qpos_weight = attn_weights[1]
    top_attn_tokens = attn_weights[2 : 2 + 300]
    wrist_attn_tokens = attn_weights[2 + 300 : 2 + 600]

    top_attn_2d = top_attn_tokens.reshape(15, 20)
    wrist_attn_2d = wrist_attn_tokens.reshape(15, 20)

    total_vis_weight = top_attn_tokens.sum() + wrist_attn_tokens.sum()
    total_prop_weight = z_weight + qpos_weight
    all_sum = total_vis_weight + total_prop_weight

    # 균등 분포 기준: 602개 토큰 중 시각 600개(99.67%), 관절 2개(0.33%)
    # 따라서 단순 합산 대신 '균등 대비 어텐션 밀도 배수(Attention Density Ratio)' 산출
    # 1.0x = 균등 분배 수준, > 1.0x = 해당 토큰들에 평균 이상의 집중 투입
    uniform_weight = 1.0 / 602.0
    top_density = (top_attn_tokens.mean() / uniform_weight)
    wrist_density = (wrist_attn_tokens.mean() / uniform_weight)
    qpos_density = (qpos_weight / uniform_weight)
    z_density = (z_weight / uniform_weight)

    # 엔트로피 (낮을수록 특정 영역에 강하게 몰입)
    # 완전 균등 시 최대 엔트로피 = ln(602) = 6.40
    eps = 1e-12
    norm_attn = attn_weights / (attn_weights.sum() + eps)
    entropy = -np.sum(norm_attn * np.log(norm_attn + eps))
    max_entropy = np.log(602)
    entropy_ratio = entropy / max_entropy  # 1.0에 가까우면 완전 분산, 낮을수록 뾰족한 집중

    # 피크 대비 평균 비율 (Peak-to-Average Ratio: 최고점이 평균의 몇 배인가)
    top_par = top_attn_tokens.max() / (top_attn_tokens.mean() + eps)
    wrist_par = wrist_attn_tokens.max() / (wrist_attn_tokens.mean() + eps)

    # 손목 카메라 중앙 ROI 집중도 (rows 3:12, cols 4:16 = 중앙 9x12 패치, 공/그리퍼 위치)
    wrist_center_roi = wrist_attn_2d[3:12, 4:16].sum() / (wrist_attn_tokens.sum() + eps) * 100.0

    # 8. 4단계 시각화 패널 생성
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"ACT Vision Pipeline Deep Inspection [Ep: {os.path.basename(hdf5_path)} | Step: {step}/{total_steps} | {query_desc}]",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    cams = [
        ("Top Camera (Workspace View)", raw_images["top"], top_energy, top_attn_2d, top_density, top_par),
        ("Wrist Camera (Gripper Close-up)", raw_images["wrist"], wrist_energy, wrist_attn_2d, wrist_density, wrist_par),
    ]

    for row_idx, (cam_title, raw_img, cnn_feat, attn_map, density, par) in enumerate(cams):
        img_h, img_w = raw_img.shape[:2]
        grid_h, grid_w = attn_map.shape  # (15, 20)

        # 패널 1: 원본 영상
        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_title(f"1. Raw RGB Image\n[{cam_title}]", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 패널 2: ResNet18 피쳐맵 활성화 (CNN 특징 검출)
        overlay_feat, _ = overlay_heatmap_on_image(raw_img, cnn_feat, colormap=cv2.COLORMAP_VIRIDIS, alpha=0.5)
        axes[row_idx, 1].imshow(overlay_feat)
        axes[row_idx, 1].set_title("2. ResNet18 Feature Map\n[Edge & Object Activation]", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # 패널 3: Transformer 크로스 어텐션 맵 (시선 히트맵)
        overlay_attn, _ = overlay_heatmap_on_image(raw_img, attn_map, colormap=cv2.COLORMAP_JET, alpha=0.6)
        axes[row_idx, 2].imshow(overlay_attn)
        axes[row_idx, 2].set_title(f"3. Transformer Attention\n[Density: {density:.2f}x (Peak PAR: {par:.1f}x)]", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # 패널 4: 모델 시선 최고 주목점 (정밀 픽셀 스케일링 변환)
        peak_row, peak_col = np.unravel_index(np.argmax(attn_map), attn_map.shape)
        # 15x20 그리드 중심을 640x480 실제 픽셀 좌표로 스케일링
        pixel_x = int((peak_col + 0.5) * (img_w / grid_w))
        pixel_y = int((peak_row + 0.5) * (img_h / grid_h))

        crosshair_img = raw_img.copy()
        # 원본 이미지 위 올바른 픽셀 좌표에 십자선 마커 표시
        cv2.drawMarker(crosshair_img, (pixel_x, pixel_y), (0, 255, 255), cv2.MARKER_CROSS, 45, 3)
        cv2.circle(crosshair_img, (pixel_x, pixel_y), 32, (0, 0, 255), 2)
        coord_text = f"TARGET Grid({peak_col},{peak_row}) -> Px({pixel_x},{pixel_y})"
        cv2.putText(
            crosshair_img,
            coord_text,
            (max(15, pixel_x - 140), max(30, pixel_y - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )
        axes[row_idx, 3].imshow(crosshair_img)
        axes[row_idx, 3].set_title(f"4. Peak Focus Crosshair\n[Grid: ({peak_col}, {peak_row}) | Pixel: ({pixel_x}, {pixel_y})]", fontsize=11, fontweight="bold")
        axes[row_idx, 3].axis("off")

    # 9. 하단 분석 진단 박스 (정량적 신뢰 지표)
    entropy_status = "Sharp Focus" if entropy_ratio < 0.85 else "Diffuse / Unfocused"
    roi_status = "Good (Center Focused)" if wrist_center_roi >= 40.0 else "Weak (Edge Dispersed)"
    
    diagnosis_text = (
        f"ATTENTION DENSITY METRICS (vs Uniform 1.0x): Top Cam={top_density:.2f}x | Wrist Cam={wrist_density:.2f}x | "
        f"qpos={qpos_density:.2f}x | Latent z={z_density:.2f}x\n"
        f"SHARPNESS & ROI: Attention Entropy={entropy_ratio*100:.1f}% ({entropy_status}) | "
        f"Wrist Peak Ratio={wrist_par:.1f}x | Wrist Center(Ball/Gripper ROI) Focus={wrist_center_roi:.1f}% ({roi_status})"
    )

    box_edge = "#28a745" if (wrist_center_roi >= 40.0 and entropy_ratio < 0.90) else "#ffc107"

    plt.figtext(
        0.5,
        0.02,
        diagnosis_text,
        wrap=True,
        horizontalalignment="center",
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor=box_edge, linewidth=2),
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)

    print("\n" + "=" * 70)
    print(f" 🎉 [비전 피쳐 & 어텐션 정밀 진단 리포트 생성 완료]")
    print(f" • 에피소드     : {os.path.basename(hdf5_path)} (Step: {step}/{total_steps})")
    print(f" • 분석 쿼리   : {query_desc}")
    print(f" • 어텐션 밀도 : Top={top_density:.2f}x, Wrist={wrist_density:.2f}x, qpos={qpos_density:.2f}x")
    print(f" • 엔트로피    : {entropy_ratio*100:.1f}% ({entropy_status})")
    print(f" • 손목 ROI    : 중앙 집중도 {wrist_center_roi:.1f}% ({roi_status})")
    print(f" 🖼️ 저장 파일   : {output_path}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SO-101 ACT 비전 피쳐맵 및 어텐션 정밀 검증 도구 (개선판)")
    parser.add_argument("--episode", type=int, default=0, help="분석할 에피소드 인덱스 (기본값: 0)")
    parser.add_argument("--step", type=int, default=None, help="분석할 프레임 스텝 (미지정 시 공 접근 중간 프레임 자동 선택)")
    parser.add_argument("--data-file", type=str, default=None, help="특정 HDF5 파일 직접 지정")
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT_DIR, "policy_best.ckpt"), help="모델 체크포인트")
    parser.add_argument("--query", type=str, default="all", help="분석할 액션 쿼리 ('all': 50개 청크 평균, '0'~'49': 특정 스텝)")
    parser.add_argument("--output", type=str, default="vision_inspection_corrected.png", help="저장할 고해상도 리포트 PNG 경로")
    args = parser.parse_args()

    hdf5_file = args.data_file
    if hdf5_file is None:
        hdf5_file = f"data/so101_pick_place/episode_{args.episode}.hdf5"

    inspect_vision_pipeline(
        hdf5_path=hdf5_file,
        step=args.step,
        ckpt_path=args.ckpt,
        query=args.query,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
