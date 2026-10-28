"""SO-101 DAPIER-ACT 멀티모달 모델 아키텍처.

모달리티 구성:
- Head: OS30A 뎁스 기반 DP3 PointNet 글로벌 인코더 (1개 토큰, 1x256)
- Wrists: 독립된 2개의 ResNet-18 (각 80개 토큰, 총 160개 토큰)
- Qpos: 12개 관절 모터 상태 (1개 토큰, 1x256)
- Latent: CVAE 잠재 변수 (1개 토큰, 1x256)
- Transformer: 163개 컨텍스트 토큰 -> 20개 학습 쿼리 -> 20스텝 x 12 액션 청크
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops.misc import FrozenBatchNorm2d

from .config import DapierACTConfig
from .pointcloud import linspace_candidates, farthest_point_sample, limits_normalize


class PositionEmbeddingSine2D(nn.Module):
    """DETR / ACT 스타일의 2D 사인파 위치 인코딩 (파라미터 없음)"""

    def __init__(
        self,
        hidden_dim: int = 256,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = 2.0 * math.pi,
    ):
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError(f"hidden_dim은 4의 배수여야 합니다. 현재값: {hidden_dim}")
        self.hidden_dim = hidden_dim
        self.num_pos_feats = hidden_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """feature_map: [B, C, H, W] -> 출력: [B, H*W, hidden_dim]"""
        b, _, h, w = feature_map.shape
        device = feature_map.device
        dtype = feature_map.dtype

        y_embed = torch.arange(1, h + 1, device=device, dtype=dtype).view(1, h, 1).expand(b, h, w)
        x_embed = torch.arange(1, w + 1, device=device, dtype=dtype).view(1, 1, w).expand(b, h, w)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos.flatten(1, 2)  # [B, H*W, hidden_dim]


class WristResNet18Encoder(nn.Module):
    """손목 카메라용 ResNet-18 인코더.
    RGB [B, 3, 240, 320] -> Layer4 [B, 512, 8, 10] -> 1x1 Conv [B, 256, 8, 10] -> 80개 토큰 [B, 80, 256]
    """

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine2D(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """image: [B, 3, H, W] in [0, 1] -> tokens [B, 80, 256], positions [B, 80, 256]"""
        features = self.backbone(image)
        features = self.input_proj(features)  # [B, 256, 8, 10]

        tokens = features.flatten(2).transpose(1, 2)  # [B, 80, 256]
        positions = self.position_embedding(features)  # [B, 80, 256]
        return tokens, positions


class DP3PointGlobalEncoder(nn.Module):
    """DP3 기반의 3D PointNet 글로벌 인코더.
    512개 점 [B, 512, 3] -> limits 정규화 -> MLP (3->64->128->256) -> MaxPool -> 1개 글로벌 토큰 [B, 1, 256]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_points: int = 512,
        candidate_points: int = 8192,
        point_min: Optional[List[float]] = None,
        point_max: Optional[List[float]] = None,
        range_eps: float = 1e-4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.candidate_points = candidate_points
        self.range_eps = range_eps

        def_min = torch.tensor(point_min if point_min else [0.15, -0.26, 0.005], dtype=torch.float32)
        def_max = torch.tensor(point_max if point_max else [0.32, 0.26, 0.40], dtype=torch.float32)

        self.register_buffer("point_min", def_min.clone())
        self.register_buffer("point_max", def_max.clone())

        self.feature_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def set_point_limits(self, p_min: Union[List[float], torch.Tensor], p_max: Union[List[float], torch.Tensor]):
        t_min = torch.as_tensor(p_min, dtype=self.point_min.dtype, device=self.point_min.device)
        t_max = torch.as_tensor(p_max, dtype=self.point_max.dtype, device=self.point_max.device)
        self.point_min.copy_(t_min)
        self.point_max.copy_(t_max)

    def forward(self, pointclouds: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """입력: [B, 512, 3] 텐서 또는 점군 텐서 리스트 -> 글로벌 토큰 [B, 1, 256], 샘플 점 [B, 512, 3]"""
        if isinstance(pointclouds, list):
            token_list = []
            xyz_list = []
            for pc in pointclouds:
                xyz = pc[:, :3]
                if xyz.shape[0] != self.num_points:
                    cand = linspace_candidates(xyz, self.candidate_points)
                    sampled = farthest_point_sample(cand, self.num_points)
                else:
                    sampled = xyz

                norm_xyz = limits_normalize(sampled, self.point_min, self.point_max, self.range_eps)
                feats = self.feature_mlp(norm_xyz)  # [512, 256]
                glob = torch.max(feats, dim=0).values.unsqueeze(0)  # [1, 256]

                token_list.append(glob)
                xyz_list.append(sampled)

            return torch.stack(token_list, dim=0), torch.stack(xyz_list, dim=0)

        else:
            # 배치 텐서 형태 [B, N, 3]
            b, n, _ = pointclouds.shape
            if n != self.num_points:
                sampled_batches = []
                for i in range(b):
                    cand = linspace_candidates(pointclouds[i], self.candidate_points)
                    sampled_batches.append(farthest_point_sample(cand, self.num_points))
                pointclouds = torch.stack(sampled_batches, dim=0)

            norm_xyz = limits_normalize(pointclouds, self.point_min, self.point_max, self.range_eps)
            feats = self.feature_mlp(norm_xyz)  # [B, 512, 256]
            glob = torch.max(feats, dim=1).values.unsqueeze(1)  # [B, 1, 256]
            return glob, pointclouds


class DapierACTMultimodal(nn.Module):
    """SO-101 양팔 12 DOF 멀티모달 액션 청킹 트랜스포머 (DAPIER-ACT)"""

    HEAD_SOURCE = 0
    LEFT_SOURCE = 1
    RIGHT_SOURCE = 2

    def __init__(self, config: Optional[DapierACTConfig] = None):
        super().__init__()
        self.cfg = config if config is not None else DapierACTConfig()

        self.qpos_dim = self.cfg.state_dim
        self.action_dim = self.cfg.action_dim
        self.chunk_size = self.cfg.chunk_size
        self.hidden_dim = self.cfg.hidden_dim
        self.latent_dim = self.cfg.latent_dim

        # ----------------------------------------------------
        # 1. 비전 및 포인트 인코더
        # ----------------------------------------------------
        self.point_encoder = DP3PointGlobalEncoder(
            hidden_dim=self.hidden_dim,
            num_points=self.cfg.num_point_samples,
            candidate_points=self.cfg.candidate_points,
            point_min=list(self.cfg.default_point_min),
            point_max=list(self.cfg.default_point_max),
        )

        if self.cfg.shared_wrist_backbone:
            shared_wrist = WristResNet18Encoder(
                hidden_dim=self.hidden_dim, pretrained=self.cfg.pretrained_wrist_backbone
            )
            self.left_wrist_encoder = shared_wrist
            self.right_wrist_encoder = shared_wrist
        else:
            self.left_wrist_encoder = WristResNet18Encoder(
                hidden_dim=self.hidden_dim, pretrained=self.cfg.pretrained_wrist_backbone
            )
            self.right_wrist_encoder = WristResNet18Encoder(
                hidden_dim=self.hidden_dim, pretrained=self.cfg.pretrained_wrist_backbone
            )

        # 센서 Identity 임베딩: 0=Head PCD, 1=Left Wrist, 2=Right Wrist
        self.source_embedding = nn.Embedding(3, self.hidden_dim)
        nn.init.normal_(self.source_embedding.weight, mean=0.0, std=0.02)

        # 모달리티 임베딩: 0=Latent, 1=Qpos, 2=Point
        self.modality_embedding = nn.Embedding(3, self.hidden_dim)
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.02)

        # ----------------------------------------------------
        # 2. Qpos & Latent 투영기
        # ----------------------------------------------------
        self.qpos_proj = nn.Linear(self.qpos_dim, self.hidden_dim)
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        # ----------------------------------------------------
        # 3. CVAE 인코더 (학습 시 액션 궤적 -> Latent 변수 z)
        # ----------------------------------------------------
        self.cvae_cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.cvae_action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.cvae_qpos_proj = nn.Linear(self.qpos_dim, self.hidden_dim)

        cvae_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.cfg.nheads,
            dim_feedforward=self.cfg.ff_dim,
            dropout=self.cfg.dropout,
            activation="relu",
            batch_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, num_layers=self.cfg.cvae_encoder_layers)
        self.cvae_mu_head = nn.Linear(self.hidden_dim, self.latent_dim)
        self.cvae_logvar_head = nn.Linear(self.hidden_dim, self.latent_dim)

        # ----------------------------------------------------
        # 4. ACT 메인 트랜스포머 (컨텍스트 인코더 & 액션 디코더)
        # ----------------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.cfg.nheads,
            dim_feedforward=self.cfg.ff_dim,
            dropout=self.cfg.dropout,
            activation="relu",
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.cfg.nheads,
            dim_feedforward=self.cfg.ff_dim,
            dropout=self.cfg.dropout,
            activation="relu",
            batch_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(dec_layer, num_layers=self.cfg.decoder_layers)

        # 20개의 학습 가능한 액션 쿼리 임베딩
        self.action_queries = nn.Embedding(self.chunk_size, self.hidden_dim)

        # 액션 출력 헤드 (256 -> 12 관절)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

    def encode_latent(
        self, qpos: torch.Tensor, actions: torch.Tensor, is_pad: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """학습 시 CVAE 잠재 변수 z 샘플링"""
        b = qpos.shape[0]
        cls_tokens = self.cvae_cls_token.expand(b, -1, -1)
        qpos_token = self.cvae_qpos_proj(qpos).unsqueeze(1)
        action_tokens = self.cvae_action_proj(actions)

        # 시퀀스: [CLS, QPOS, Action_1, ..., Action_20]
        seq = torch.cat([cls_tokens, qpos_token, action_tokens], dim=1)

        mask = None
        if is_pad is not None:
            # CLS(False) + QPOS(False) + is_pad
            prefix_mask = torch.zeros(b, 2, dtype=torch.bool, device=qpos.device)
            mask = torch.cat([prefix_mask, is_pad], dim=1)

        enc_out = self.cvae_encoder(seq, src_key_padding_mask=mask)
        cls_out = enc_out[:, 0]

        mu = self.cvae_mu_head(cls_out)
        logvar = self.cvae_logvar_head(cls_out)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(
        self,
        qpos: torch.Tensor,
        pointcloud: Union[torch.Tensor, List[torch.Tensor]],
        left_wrist: torch.Tensor,
        right_wrist: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """멀티모달 순전파.

        Args:
            qpos: [B, 12] 관절 각도
            pointcloud: [B, 512, 3] 미터 단위 작업영역 포인트
            left_wrist: [B, 3, 240, 320] RGB in [0, 1]
            right_wrist: [B, 3, 240, 320] RGB in [0, 1]
            actions: [B, 20, 12] (학습 시 정답 액션 청크)
            is_pad: [B, 20] (패딩 마스크)
        """
        b = qpos.shape[0]

        # 1. CVAE Latent
        if actions is not None:
            latent, mu, logvar = self.encode_latent(qpos, actions[:, : self.chunk_size], is_pad)
        else:
            latent = torch.zeros(b, self.latent_dim, dtype=qpos.dtype, device=qpos.device)
            mu, logvar = None, None

        # 2. Head PointCloud 토큰 (1개 토큰: [B, 1, 256])
        point_feat, sampled_xyz = self.point_encoder(pointcloud)
        head_token = (
            point_feat
            + self.modality_embedding.weight[2][None, None, :]
            + self.source_embedding.weight[self.HEAD_SOURCE][None, None, :]
        )

        # 3. Left Wrist RGB 토큰 (80개 토큰: [B, 80, 256])
        left_tok, left_pos = self.left_wrist_encoder(left_wrist)
        left_token = left_tok + left_pos + self.source_embedding.weight[self.LEFT_SOURCE][None, None, :]

        # 4. Right Wrist RGB 토큰 (80개 토큰: [B, 80, 256])
        right_tok, right_pos = self.right_wrist_encoder(right_wrist)
        right_token = right_tok + right_pos + self.source_embedding.weight[self.RIGHT_SOURCE][None, None, :]

        # 5. Qpos & Latent 토큰
        qpos_token = self.qpos_proj(qpos).unsqueeze(1) + self.modality_embedding.weight[1][None, None, :]
        latent_token = self.latent_out_proj(latent).unsqueeze(1) + self.modality_embedding.weight[0][None, None, :]

        # 6. 총 163개 멀티모달 컨텍스트 결합 [B, 163, 256]
        # 순서: Latent(1) + Qpos(1) + Head(1) + Left(80) + Right(80)
        context = torch.cat([latent_token, qpos_token, head_token, left_token, right_token], dim=1)

        # 7. 트랜스포머 인코더 및 디코더
        memory = self.context_encoder(context)  # [B, 163, 256]

        queries = self.action_queries.weight.unsqueeze(0).expand(b, -1, -1)  # [B, 20, 256]
        decoded = self.action_decoder(queries, memory)  # [B, 20, 256]

        action_hat = self.action_head(decoded)  # [B, 20, 12]

        if return_features:
            return action_hat, mu, logvar, {"context": context, "memory": memory, "sampled_xyz": sampled_xyz}

        return action_hat, mu, logvar


def dapier_act_loss(
    action_pred: torch.Tensor,
    action_gt: torch.Tensor,
    is_pad: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
    kl_weight: float = 10.0,
) -> Dict[str, torch.Tensor]:
    """L1 액션 복원 손실 + CVAE KL Divergence 손실 계산"""
    if is_pad is not None:
        valid_mask = (~is_pad).unsqueeze(-1).float()
        l1_diff = torch.abs(action_pred - action_gt) * valid_mask
        l1_loss = l1_diff.sum() / valid_mask.sum().clamp_min(1.0)
    else:
        l1_loss = F.l1_loss(action_pred, action_gt)

    if mu is not None and logvar is not None:
        # KL(q(z|x, a) || p(z)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl.mean()
    else:
        kl_loss = torch.tensor(0.0, device=action_pred.device)

    total_loss = l1_loss + kl_weight * kl_loss
    return {"loss": total_loss, "l1": l1_loss, "kl": kl_loss}
