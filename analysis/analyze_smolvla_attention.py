"""SO-101 SmolVLA 비전 피처맵 및 언어-시각 어텐션 정밀 분석 도구.

로봇 팔을 직접 움직이지 않고, 모델의 뇌(신경망)가:
  1) 비전 백본(SigLIP)에서 공의 시각적 형태/색상을 뚜렷이 인지하는지 (Feature Map)
  2) 언어 프롬프트("green" vs "pink")에 따라 시선(Attention)이 각 공으로 이동하는지
  3) 예측 액션 궤적(Action Chunk)이 두 공 방향으로 선명하게 분기(Branching)하는지
를 1:1 대조 시각화하고 정량 분석합니다.
"""

import argparse
import glob
import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from hardware.camera_interface import MultiCameraCapture, auto_detect_camera_indices
except ImportError:
    MultiCameraCapture = None
    auto_detect_camera_indices = None

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def resolve_checkpoint_path(ckpt_arg: str) -> str:
    """체크포인트 스텝 번호(예: 5000, 10000, 20000, last) 또는 경로 해석"""
    if os.path.isdir(ckpt_arg):
        return ckpt_arg

    for cand_dir in ["smolvla_two_balls_lora", "smolvla_two_balls_100ep", "smolvla_two_balls", "smolvla_so101"]:
        ckpt_base = os.path.join(PROJECT_ROOT, "ckpt", cand_dir, "checkpoints")
        if os.path.exists(ckpt_base):
            break

    if ckpt_arg.isdigit():
        padded = f"{int(ckpt_arg):06d}"
        path = os.path.join(ckpt_base, padded, "pretrained_model")
    elif ckpt_arg in ["last", "final"]:
        path = os.path.join(ckpt_base, "last", "pretrained_model")
    else:
        path = os.path.join(ckpt_base, ckpt_arg, "pretrained_model")

    if not os.path.isdir(path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {path}")
    return path


def load_input_images(source: str, hdf5_path: str = None, frame_idx: int = 15) -> Tuple[np.ndarray, np.ndarray, str]:
    """분석 대상 이미지 로드 (live 카메라 또는 hdf5 데이터셋)"""
    top_img, wrist_img = None, None
    desc = ""

    if source == "live":
        print("\n📸 [라이브 카메라 캡처] Top 및 Wrist 카메라에서 1프레임을 가져옵니다...")
        if auto_detect_camera_indices is None:
            raise RuntimeError("camera_interface 모듈을 불러올 수 없습니다.")
        top_id, wrist_id = auto_detect_camera_indices()
        mc = MultiCameraCapture({"top": top_id, "wrist": wrist_id}, width=640, height=480, fps=30)
        mc.start()
        # 안정적인 노출/화이트밸런스를 위해 5프레임 워밍업
        for _ in range(5):
            time.sleep(0.05)
            frames = mc.read_all()
        top_bgr = frames.get("top")
        wrist_bgr = frames.get("wrist")
        mc.stop()

        if top_bgr is None or wrist_bgr is None:
            raise RuntimeError("라이브 카메라에서 프레임을 읽지 못했습니다.")

        top_img = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
        wrist_img = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
        desc = f"Live Cameras (Top: /dev/video{top_id}, Wrist: /dev/video{wrist_id})"

    else:
        if not hdf5_path or not os.path.exists(hdf5_path):
            candidates = [
                os.path.join(PROJECT_ROOT, "data", "so101_pick_green", "episode_0.hdf5"),
                os.path.join(PROJECT_ROOT, "data", "so101_pick_pink", "episode_0.hdf5"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    hdf5_path = c
                    break
        if not hdf5_path or not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"분석할 HDF5 파일이 없습니다: {hdf5_path}")

        print(f"\n📂 [HDF5 데이터셋 로드] {hdf5_path} (Frame: {frame_idx})")
        with h5py.File(hdf5_path, "r") as f:
            top_seq = f["observations/images/top"]
            wrist_seq = f["observations/images/wrist"]
            idx = min(frame_idx, len(top_seq) - 1)
            top_img = top_seq[idx]
            wrist_img = wrist_seq[idx]
        desc = f"HDF5 Dataset ({os.path.basename(hdf5_path)} - Frame #{idx})"

    return top_img, wrist_img, desc


class SmolVLAAnalyzer:
    """SmolVLA 비전 피처 및 크로스 어텐션 분석기"""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"\n🧠 [모델 로드] 체크포인트 로드 중: {checkpoint_path} (Device: {self.device})")

        if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            from peft import PeftModel, PeftConfig
            print("📦 [PEFT] LoRA 어댑터 감지됨. 베이스 모델과 어댑터를 함께 로드합니다...")
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            base_policy = SmolVLAPolicy.from_pretrained(peft_config.base_model_name_or_path)
            self.policy = PeftModel.from_pretrained(base_policy, checkpoint_path)
        else:
            self.policy = SmolVLAPolicy.from_pretrained(checkpoint_path)

        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=checkpoint_path
        )
        print("✅ SmolVLA 모델 및 프로세서 준비 완료.")

    def get_vla_model(self):
        raw = self.policy.get_base_model() if hasattr(self.policy, "get_base_model") else self.policy
        return raw.model

    def extract_vision_feature_maps(self, top_rgb: np.ndarray, wrist_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SigLIP 비전 백본의 마지막 레이어 활성화 맵(Feature Map) 추출"""
        vla_model = self.get_vla_model()
        vlm = vla_model.vlm_with_expert.vlm
        vision_model = vlm.model.vision_model
        device = next(self.policy.parameters()).device
        dtype = vision_model.embeddings.patch_embedding.weight.dtype

        # SmolVLA 입력 규격: [0, 1] float32, (C, H, W)
        top_t = torch.from_numpy(top_rgb).permute(2, 0, 1).float() / 255.0
        wrist_t = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float() / 255.0

        obs = {
            "observation.state": torch.zeros(6),
            "observation.images.top": top_t,
            "observation.images.wrist": wrist_t,
            "task": "dummy task",
        }
        processed = self.preprocessor(obs)

        with torch.no_grad():
            images, img_masks = self.policy.prepare_images(processed)
            img_top = images[0].to(device=device, dtype=dtype)
            img_wrist = images[1].to(device=device, dtype=dtype)

            top_out = vision_model(img_top)[0]  # (1, num_patches, 768)
            wrist_out = vision_model(img_wrist)[0]

            top_norm = torch.norm(top_out.float(), dim=-1).squeeze(0).cpu().numpy()
            wrist_norm = torch.norm(wrist_out.float(), dim=-1).squeeze(0).cpu().numpy()

            grid_top = int(round(len(top_norm) ** 0.5))
            grid_wrist = int(round(len(wrist_norm) ** 0.5))

            patch_grid_top = top_norm.reshape(grid_top, grid_top)
            patch_grid_wrist = wrist_norm.reshape(grid_wrist, grid_wrist)

        # [0, 1] 정규화
        def norm_map(m):
            return (m - m.min()) / (m.max() - m.min() + 1e-8)

        map_top = norm_map(patch_grid_top)
        map_wrist = norm_map(patch_grid_wrist)

        # 480x640 크기로 업샘플링
        map_top_resized = cv2.resize(map_top, (640, 480), interpolation=cv2.INTER_CUBIC)
        map_wrist_resized = cv2.resize(map_wrist, (640, 480), interpolation=cv2.INTER_CUBIC)

        return map_top_resized, map_wrist_resized

    def analyze_attention_and_action(
        self, top_rgb: np.ndarray, wrist_rgb: np.ndarray, task_prompt: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """특정 언어 지시문에 대한 시선 어텐션 맵 및 50스텝 액션 청크 추출"""
        top_t = torch.from_numpy(top_rgb).permute(2, 0, 1).float() / 255.0
        wrist_t = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float() / 255.0

        obs = {
            "observation.state": torch.zeros(6),
            "observation.images.top": top_t,
            "observation.images.wrist": wrist_t,
            "task": task_prompt,
        }
        processed = self.preprocessor(obs)

        with torch.no_grad():
            self.policy.reset()
            # 50스텝 예측 액션 청크
            action_chunk = self.policy.predict_action_chunk(processed)
            action_unnorm = self.postprocessor(action_chunk)  # (1, 50, 6)

        actions_np = action_unnorm.squeeze(0).cpu().numpy()  # (50, 6)

        # 프롬프트 토큰과 비전 토큰 간의 Attention 유사도 맵 생성
        images, _ = self.policy.prepare_images(processed)
        vla_model = self.get_vla_model()
        vlm = vla_model.vlm_with_expert.vlm
        device = next(self.policy.parameters()).device
        dtype = vlm.model.vision_model.embeddings.patch_embedding.weight.dtype

        with torch.no_grad():
            img_top = images[0].to(device=device, dtype=dtype)
            img_wrist = images[1].to(device=device, dtype=dtype)
            top_vis_out = vlm.model.vision_model(img_top)[0]  # (1, num_patches, 768)
            wrist_vis_out = vlm.model.vision_model(img_wrist)[0]

            # VLM 커넥터(Connector)를 통해 비전 특징을 텍스트 차원(960)으로 투영
            conn_dtype = next(vlm.model.connector.parameters()).dtype
            top_proj = vlm.model.connector(top_vis_out.to(dtype=conn_dtype)).float()  # (1, tokens, 960)
            wrist_proj = vlm.model.connector(wrist_vis_out.to(dtype=conn_dtype)).float()

            lang_tokens = processed["observation.language.tokens"].to(device=device)  # (1, 48)
            text_embeds = vlm.model.text_model.embed_tokens(lang_tokens).float()  # (1, 48, 960)

            # 핵심 키워드(green vs pink) 토큰 인덱스 찾기
            tok = vla_model.vlm_with_expert.processor.tokenizer
            decoded_tokens = [tok.decode([t.item()]).strip().lower() for t in lang_tokens[0]]

            target_kw = "green" if "green" in task_prompt.lower() else "pink"
            kw_idx = -1
            for i, dt in enumerate(decoded_tokens):
                if target_kw in dt:
                    kw_idx = i
                    break
            if kw_idx == -1:
                kw_idx = 2  # fallback

            # 텍스트 키워드 벡터 (1, 960)
            kw_vec = F.normalize(text_embeds[:, kw_idx, :], dim=-1)

            # 비전 토큰들과의 코사인 유사도 맵 계산 (tokens,)
            top_sim = torch.matmul(F.normalize(top_proj, dim=-1), kw_vec.unsqueeze(-1)).squeeze()
            wrist_sim = torch.matmul(F.normalize(wrist_proj, dim=-1), kw_vec.unsqueeze(-1)).squeeze()

            g_top = int(round(len(top_sim) ** 0.5))
            g_wrist = int(round(len(wrist_sim) ** 0.5))

            top_sim_grid = top_sim.view(g_top, g_top).cpu().numpy()
            wrist_sim_grid = wrist_sim.view(g_wrist, g_wrist).cpu().numpy()

            def norm_att(g):
                g = np.maximum(g, 0)
                return (g - g.min()) / (g.max() - g.min() + 1e-8)

            top_att_2d = norm_att(top_sim_grid)
            wrist_att_2d = norm_att(wrist_sim_grid)

            top_att_res = cv2.resize(top_att_2d, (640, 480), interpolation=cv2.INTER_CUBIC)
            wrist_att_res = cv2.resize(wrist_att_2d, (640, 480), interpolation=cv2.INTER_CUBIC)

        return top_att_res, wrist_att_res, actions_np


def render_heatmap_overlay(rgb_img: np.ndarray, heatmap_norm: np.ndarray, colormap=cv2.COLORMAP_JET, alpha=0.55) -> np.ndarray:
    """히트맵을 원본 RGB 이미지 위에 반투명 오버레이"""
    hm_uint8 = np.uint8(np.clip(heatmap_norm * 255.0, 0, 255))
    colored_hm = cv2.applyColorMap(hm_uint8, colormap)
    colored_hm_rgb = cv2.cvtColor(colored_hm, cv2.COLOR_BGR2RGB)

    blended = np.uint8(rgb_img * (1.0 - alpha) + colored_hm_rgb * alpha)
    return blended


def main():
    parser = argparse.ArgumentParser(description="SmolVLA 비전 피처맵 및 언어-시각 어텐션 정밀 분석")
    parser.add_argument("--checkpoint", type=str, default="last", help="체크포인트 스텝 (예: 5000, 10000, 20000, last)")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "live", "hdf5"], help="입력 이미지 소스")
    parser.add_argument("--hdf5-path", type=str, default="", help="HDF5 파일 경로")
    parser.add_argument("--frame-idx", type=int, default=15, help="HDF5 내 프레임 인덱스")
    parser.add_argument("--output-dir", type=str, default="analysis/outputs", help="시각화 저장 폴더")
    parser.add_argument("--device", type=str, default="cuda", help="실행 디바이스")
    args = parser.parse_args()

    # 소스 자동 결정
    source = args.source
    if source == "auto":
        # 카메라 2개가 다 열려있으면 live, 아니면 hdf5
        source = "live" if (os.path.exists("/dev/video4") and (os.path.exists("/dev/video6") or os.path.exists("/dev/video7"))) else "hdf5"

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    step_label = os.path.basename(os.path.dirname(ckpt_path))

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(" 🔍 [SO-101 SmolVLA 비전 피처맵 및 어텐션 정밀 분석기]")
    print(f" • 체크포인트 : {ckpt_path} (Step: {step_label})")
    print(f" • 이미지 소스 : {source.upper()}")
    print(f" • 저장 폴더   : {args.output_dir}")
    print("=" * 70)

    # 1. 이미지 로드
    top_rgb, wrist_rgb, src_desc = load_input_images(source, args.hdf5_path, args.frame_idx)

    # 2. 분석기 초기화
    analyzer = SmolVLAAnalyzer(ckpt_path, device=args.device)

    # 3. 비전 피처맵 추출
    print("\n1️⃣ [비전 백본 피처맵 추출 중...] SigLIP 패치 활성화 강도 계산...")
    top_feat_map, wrist_feat_map = analyzer.extract_vision_feature_maps(top_rgb, wrist_rgb)

    # 4. 언어별 어텐션 및 예측 액션 추출
    prompt_green = "Pick up the green ball and place it in the box"
    prompt_pink = "Pick up the pink ball and place it in the box"

    print(f"2️⃣ [어텐션 추출] 지시문 A: \"{prompt_green}\"")
    top_att_green, wrist_att_green, act_green = analyzer.analyze_attention_and_action(top_rgb, wrist_rgb, prompt_green)

    print(f"3️⃣ [어텐션 추출] 지시문 B: \"{prompt_pink}\"")
    top_att_pink, wrist_att_pink, act_pink = analyzer.analyze_attention_and_action(top_rgb, wrist_rgb, prompt_pink)

    # 5. 히트맵 오버레이 렌더링
    top_feat_overlay = render_heatmap_overlay(top_rgb, top_feat_map, cv2.COLORMAP_VIRIDIS)
    wrist_feat_overlay = render_heatmap_overlay(wrist_rgb, wrist_feat_map, cv2.COLORMAP_VIRIDIS)

    top_green_overlay = render_heatmap_overlay(top_rgb, top_att_green, cv2.COLORMAP_JET)
    wrist_green_overlay = render_heatmap_overlay(wrist_rgb, wrist_att_green, cv2.COLORMAP_JET)

    top_pink_overlay = render_heatmap_overlay(top_rgb, top_att_pink, cv2.COLORMAP_JET)
    wrist_pink_overlay = render_heatmap_overlay(wrist_rgb, wrist_att_pink, cv2.COLORMAP_JET)

    # 6. 종합 2x4 패널 시각화 저장
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)

    # 1행: Top Camera
    axes[0, 0].imshow(top_rgb)
    axes[0, 0].set_title(f"Top: Original RGB\n({src_desc})", fontsize=11, fontweight="bold")

    axes[0, 1].imshow(top_feat_overlay)
    axes[0, 1].set_title("Top: SigLIP Vision Feature\n(High Salience = Bright)", fontsize=11, fontweight="bold")

    axes[0, 2].imshow(top_green_overlay)
    axes[0, 2].set_title("Top: Attention on \"green ball\"\n(Red = Hotspot)", fontsize=11, fontweight="bold", color="green")

    axes[0, 3].imshow(top_pink_overlay)
    axes[0, 3].set_title("Top: Attention on \"pink ball\"\n(Red = Hotspot)", fontsize=11, fontweight="bold", color="magenta")

    # 2행: Wrist Camera
    axes[1, 0].imshow(wrist_rgb)
    axes[1, 0].set_title("Wrist: Original RGB", fontsize=11, fontweight="bold")

    axes[1, 1].imshow(wrist_feat_overlay)
    axes[1, 1].set_title("Wrist: SigLIP Vision Feature", fontsize=11, fontweight="bold")

    axes[1, 2].imshow(wrist_green_overlay)
    axes[1, 2].set_title("Wrist: Attention on \"green ball\"", fontsize=11, fontweight="bold", color="green")

    axes[1, 3].imshow(wrist_pink_overlay)
    axes[1, 3].set_title("Wrist: Attention on \"pink ball\"", fontsize=11, fontweight="bold", color="magenta")

    for ax in axes.ravel():
        ax.axis("off")

    plt.suptitle(f"SmolVLA Multimodal Attention & Vision Feature Diagnostic (Step: {step_label})", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_panel_path = os.path.join(args.output_dir, f"attention_analysis_step_{step_label}.png")
    plt.savefig(out_panel_path, bbox_inches="tight")
    plt.close()

    # 7. 액션 궤적 분기 그래프 저장 (6개 관절 궤적 비교)
    joint_names = ["1.Base (회전)", "2.Shoulder (어깨)", "3.Elbow (팔꿈치)", "4.Wrist Pitch (손목상하)", "5.Wrist Roll (손목회전)", "6.Gripper (그리퍼)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
    steps_x = np.arange(len(act_green))

    for j, ax in enumerate(axes.ravel()):
        ax.plot(steps_x, act_green[:, j], color="#16A34A", linewidth=2.2, label="\"green ball\" trajectory")
        ax.plot(steps_x, act_pink[:, j], color="#E11D48", linewidth=2.2, linestyle="--", label="\"pink ball\" trajectory")
        ax.set_title(joint_names[j], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Chunk Step (0~50)", fontsize=9)
        ax.set_ylabel("Angle (rad)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="best")

    plt.suptitle(f"Predicted Action Trajectory Divergence: Green vs Pink (Step: {step_label})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_act_path = os.path.join(args.output_dir, f"action_branching_step_{step_label}.png")
    plt.savefig(out_act_path, bbox_inches="tight")
    plt.close()

    # 8. 정량적 판정 리포트 터미널 출력
    base_divergence = np.abs(act_green[:, 0] - act_pink[:, 0]).mean()
    total_divergence = np.abs(act_green - act_pink).mean()

    print("\n" + "=" * 70)
    print(" 📊 [정밀 분석 판정 리포트]")
    print("=" * 70)
    print(f" 🖼️ 시각화 패널 저장 완료 : {out_panel_path}")
    print(f" 📈 궤적 분기 그래프 저장 : {out_act_path}")
    print("----------------------------------------------------------------------")
    print(f" • 1번 Base 관절 분기 차이 (평균) : {base_divergence:.4f} rad ({np.degrees(base_divergence):.2f}°)")
    print(f" • 전체 6개 관절 분기 차이 (평균)  : {total_divergence:.4f} rad")

    if base_divergence < 0.05:
        print("\n ⚠️ [진단 결과: 언어 무시 상태 (Language Ignored)]")
        print("    -> 초록 공과 핑크 공 지시를 바꾸었는데도 Base 관절 궤적이 거의 갈라지지 않습니다.")
        print("    -> 모델이 단어를 읽지 않고 시각적으로 가까운 1개 공으로만 가고 있습니다.")
    else:
        print("\n ✅ [진단 결과: 언어 분기 정상 작동 (Language Controlled)]")
        print(f"    -> 두 지시문에 따라 팔이 서로 다른 방향({np.degrees(base_divergence):.1f}도 차이)으로 확실히 갈라집니다.")
        print("    -> 이 경우 헛손질의 원인은 언어 인식이 아니라 '공 위치에 대한 손끝 오프셋(거리 측정)' 문제입니다.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
