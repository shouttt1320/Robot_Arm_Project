"""양안(Dual Eyes) 카메라 및 DINOv2 백본 기반 ACT (DETR-VAE) 아키텍처."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.zeros((n_position, d_hid))
    for i in range(n_position):
        sinusoid_table[i, 0::2] = np.sin(get_position_angle_vec(i)[0::2])
        sinusoid_table[i, 1::2] = np.cos(get_position_angle_vec(i)[1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


class DETRVAE(nn.Module):
    """양안 카메라 및 DINOv2 기반 SO-101 (6-DOF) 액션 청킹 CVAE 트랜스포머 모델."""

    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        encoder: nn.Module,
        state_dim: int = 6,
        num_queries: int = 50,
        camera_names: list = None,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_queries = num_queries
        self.camera_names = camera_names or ["cam_left", "cam_right"]
        self.transformer = transformer
        self.encoder = encoder
        self.latent_dim = latent_dim

        hidden_dim = transformer.d_model

        # 액션 및 패딩 예측 헤드 (6차원 관절)
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 이미지 백본 투영 (DINOv2 num_channels -> hidden_dim 512) 및 6차원 로봇 상태 투영
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # CVAE 인코더 파라미터
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
        self.register_buffer("pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))

        # 디코더 잠재 변수 투영 및 위치 임베딩
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        qpos: (bs, 6)
        image: (bs, num_cam, 3, H, W)
        actions: (bs, chunk_size, 6) - 학습 시에만 제공
        is_pad: (bs, chunk_size) - 마스크
        """
        is_training = actions is not None
        bs = qpos.shape[0]

        # 1. CVAE 인코더 (학습 시: 액션 시퀀스를 잠재 변수 z로 압축 / 추론 시: z=0)
        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)

            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad_full = torch.cat([cls_joint_is_pad, is_pad], axis=1)

            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_full)
            encoder_output = encoder_output[0]  # [CLS] 토큰 출력

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # 2. 멀티카메라(좌/우 양안) 특징 추출 및 너비(Width) 축 결합
        # 위치 임베딩도 함께 너비 축으로 이어붙여 트랜스포머 시퀀스 상에서 카메라별 위치가 명확히 구분됨
        all_cam_features = []
        all_cam_pos = []
        for cam_id in range(len(self.camera_names)):
            features, pos = self.backbone(image[:, cam_id])
            features = features[0]
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)

        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        proprio_input = self.input_proj_robot_state(qpos)

        # 3. DETR 트랜스포머 디코더 (미래 액션 청크 일괄 예측)
        hs = self.transformer(
            src, None, self.query_embed.weight, pos,
            latent_input, proprio_input, self.additional_pos_embed.weight
        )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]


def build_encoder(hidden_dim: int = 512, enc_layers: int = 4, nheads: int = 8, dim_feedforward: int = 3200):
    encoder_layer = TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        activation="relu",
    )
    return TransformerEncoder(encoder_layer, num_layers=enc_layers)


def build_act_model(
    state_dim: int = 6,
    num_queries: int = 50,
    camera_names: list = None,
    backbone_name: str = "dinov2_vits14",
    hidden_dim: int = 512,
    dim_feedforward: int = 3200,
    enc_layers: int = 4,
    dec_layers: int = 7,
    nheads: int = 8,
    latent_dim: int = 32,
    train_backbone: bool = False,
) -> DETRVAE:
    """양안 카메라 DINOv2 ACT 모델 인스턴스 생성"""
    camera_names = camera_names or ["cam_left", "cam_right"]
    backbone = build_backbone(backbone_name=backbone_name, hidden_dim=hidden_dim, train_backbone=train_backbone)
    transformer = build_transformer(
        hidden_dim=hidden_dim,
        nheads=nheads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
    )
    encoder = build_encoder(
        hidden_dim=hidden_dim,
        enc_layers=enc_layers,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
    )

    model = DETRVAE(
        backbone=backbone,
        transformer=transformer,
        encoder=encoder,
        state_dim=state_dim,
        num_queries=num_queries,
        camera_names=camera_names,
        latent_dim=latent_dim,
    )
    return model
