"""ResNet-18 피처맵(Feature Map) 시각화 및 해상도 손실 정밀 분석 도구.

ACT 모델이 카메라 영상(640x480)을 받아 ResNet18 각 레이어를 거치면서
피처맵이 어떻게 변하고, 탁구공(피사체)이 몇 픽셀로 뭉개지는지 단계별로 시각화합니다.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_and_visualize_features(img_path: str, output_png: str, cam_name: str = "Top Camera"):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {img_path}")

    # 1. 이미지 로드 및 전처리
    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = rgb_img.shape[:2]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(rgb_img).unsqueeze(0)  # (1, 3, 480, 640)

    # 2. ResNet18 백본 로드 (ACT와 동일 구조)
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()

    # 중간 레이어 피처맵 캡처 훅(Hook) 등록
    features = {}
    def hook_fn(name):
        def _hook(module, inp, out):
            features[name] = out.detach().cpu()
        return _hook

    resnet.layer1.register_forward_hook(hook_fn("layer1"))
    resnet.layer2.register_forward_hook(hook_fn("layer2"))
    resnet.layer3.register_forward_hook(hook_fn("layer3"))
    resnet.layer4.register_forward_hook(hook_fn("layer4"))

    with torch.no_grad():
        _ = resnet(input_tensor)

    # 3. 레이어별 해상도 및 통계 분석
    l1 = features["layer1"][0]  # (64, 120, 160)
    l2 = features["layer2"][0]  # (128, 60, 80)
    l3 = features["layer3"][0]  # (256, 30, 40)
    l4 = features["layer4"][0]  # (512, 15, 20) -> ACT 트랜스포머에 들어가는 최종 맵!

    layers_info = [
        ("Layer 1 (1/4 축소)", l1, 4),
        ("Layer 2 (1/8 축소)", l2, 8),
        ("Layer 3 (1/16 축소)", l3, 16),
        ("Layer 4 (1/32 축소 - ACT 최종)", l4, 32),
    ]

    # 4. 시각화 피규어 생성 (3행 4열)
    fig = plt.figure(figsize=(22, 14), facecolor="white")
    gs = fig.add_gridspec(3, 4, hspace=0.32, wspace=0.22)

    # (1) 맨 윗줄: 원본 영상 및 32x32 격자 오버레이
    # Col 0-1: 원본 영상
    ax_orig = fig.add_subplot(gs[0, 0:2])
    ax_orig.imshow(rgb_img)
    ax_orig.set_title(f"1. Original Image ({cam_name}) [640 x 480 픽셀]", fontsize=13, fontweight="bold", pad=10)
    ax_orig.axis("off")

    # Col 2-3: 15x20 ACT 최종 격자(Grid) 오버레이 (격자 한 칸 = 32x32 픽셀)
    ax_grid = fig.add_subplot(gs[0, 2:4])
    grid_img = rgb_img.copy()
    # 32x32 간격으로 선 그리기
    for x in range(0, w_orig, 32):
        cv2.line(grid_img, (x, 0), (x, h_orig), (100, 255, 100), 1)
    for y in range(0, h_orig, 32):
        cv2.line(grid_img, (0, y), (w_orig, y), (100, 255, 100), 1)
    ax_grid.imshow(grid_img)
    ax_grid.set_title("2. ACT Transformer Input Grid [15행 x 20열 | 격자 1칸 = 32x32 픽셀]", fontsize=13, fontweight="bold", pad=10)
    ax_grid.axis("off")

    # (2) 둘째 줄: 4개 레이어별 피처맵 활성화 히트맵 (Channel Mean / Energy)
    for col_idx, (title, feat, stride) in enumerate(layers_info):
        ax_heat = fig.add_subplot(gs[1, col_idx])
        c, h, w = feat.shape
        energy = torch.mean(torch.abs(feat), dim=0).numpy()
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

        im = ax_heat.imshow(energy_norm, cmap="inferno", interpolation="nearest")
        ax_heat.set_title(f"{title}\n해상도: {w}x{h} ({c}채널)", fontsize=11, fontweight="bold")
        ax_heat.set_xlabel(f"가로 {w}칸", fontsize=9)
        ax_heat.set_ylabel(f"세로 {h}칸", fontsize=9)
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # (3) 셋째 줄: 원본 영상 위에 레이어별 피처맵을 반투명 오버레이
    for col_idx, (title, feat, stride) in enumerate(layers_info):
        ax_over = fig.add_subplot(gs[2, col_idx])
        c, h, w = feat.shape
        energy = torch.mean(torch.abs(feat), dim=0).numpy()
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

        # 640x480으로 업샘플링 후 오버레이
        heat_resized = cv2.resize((energy_norm * 255).astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
        heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(rgb_img, 0.55, heat_color_rgb, 0.45, 0)
        ax_over.imshow(blended)
        ax_over.set_title(f"Overlay: {title}\n[격자 크기: {stride}x{stride}px]", fontsize=11, fontweight="bold")
        ax_over.axis("off")

    fig.suptitle(
        f"SO-101 ResNet-18 Feature Map Resolution & Downsampling Analysis ({cam_name})\n"
        f"Notice how Layer 4 (15x20) turns a 15~20px ball into a single blurred grid cell!",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 피처맵 분석 이미지 저장 완료 ➔ {output_png}")


def main():
    top_path = "logs/latest_frames/step_001_top.jpg"
    wrist_path = "logs/latest_frames/step_001_wrist.jpg"

    if os.path.exists(top_path):
        extract_and_visualize_features(top_path, "logs/feature_map_top.png", "Top Camera")
    if os.path.exists(wrist_path):
        extract_and_visualize_features(wrist_path, "logs/feature_map_wrist.png", "Wrist Camera")


if __name__ == "__main__":
    main()
