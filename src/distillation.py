# -*- coding: utf-8 -*-
"""
Temporally Shifted Distillation for CG-CRASH v4
================================================
灵感来源：ICLR 2026 "Lightweight Spatio-Temporal Modeling via
          Temporally Shifted Distillation for Real-Time Accident Anticipation"

核心思想：
  用冻结的 CLIP 图像 encoder 作为 teacher，
  让 GRU student 在时刻 t 学习预测时刻 t+1 的 CLIP 特征。
  这使 student 获得「预见未来」的能力，无需视频预训练 teacher。

与 CBM 的深度结合：
  - CLIP teacher 特征与 concept 文本嵌入在同一语义空间
  - Contrastive loss 直接拉近 concept 激活与 CLIP 文本特征
  - 形成：视频帧 → student → concept bottleneck → CLIP语义空间 的完整对齐

三个 loss 组件：
  1. L_spatial:  student 在 t 时刻的特征对齐 teacher 在 t 时刻的特征
  2. L_temporal: student 在 t 时刻的特征对齐 teacher 在 t+1 时刻的特征（核心创新）
  3. L_contrast: student 视觉特征与 concept 文本特征的对比学习
"""
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalProjectionHead(nn.Module):
    """Projects student features to teacher feature space."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TemporallyShiftedDistillation(nn.Module):
    """
    Temporally Shifted Distillation Module.

    During training:
      - Spatial loss:  align student(t) → teacher(t)
      - Temporal loss: align student(t) → teacher(t+1)  ← key innovation
      - Contrast loss: align concept activations with CLIP text embeddings

    Args:
        student_dim:   dimension of student GRU hidden states
        teacher_dim:   dimension of CLIP image features (512 for ViT-B/16)
        num_concepts:  number of CBM concepts
        lambda_spatial:  weight for spatial distillation loss
        lambda_temporal: weight for temporal shift loss
        lambda_contrast: weight for contrastive loss
        temperature:     contrastive loss temperature
    """
    def __init__(
        self,
        student_dim:     int   = 256,
        teacher_dim:     int   = 512,
        num_concepts:    int   = 837,
        lambda_spatial:  float = 1.0,
        lambda_temporal: float = 2.0,   # temporal shift is more important
        lambda_contrast: float = 0.5,
        temperature:     float = 0.07,
    ):
        super().__init__()
        self.lambda_spatial  = lambda_spatial
        self.lambda_temporal = lambda_temporal
        self.lambda_contrast = lambda_contrast
        self.temperature     = nn.Parameter(torch.tensor(temperature))
        self.student_dim     = student_dim
        self.teacher_dim     = teacher_dim

        # Spatial projection head: student_dim → teacher_dim
        self.spatial_proj = SpatioTemporalProjectionHead(student_dim, teacher_dim)

        # Temporal projection head: student(t) → teacher(t+1)
        # Has an extra temporal transform layer to learn "future prediction"
        self.temporal_proj = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.LayerNorm(student_dim),
            nn.GELU(),
            nn.Linear(student_dim, student_dim),  # temporal transform
            nn.GELU(),
        )
        self.temporal_head = SpatioTemporalProjectionHead(student_dim, teacher_dim)

        # Concept-to-CLIP alignment: concept activations → CLIP text space
        # Aligns concept bottleneck with CLIP's semantic space
        self.concept_proj = SpatioTemporalProjectionHead(num_concepts, teacher_dim)

    def spatial_loss(
        self,
        student_feats: torch.Tensor,   # (B, T, student_dim)
        teacher_feats: torch.Tensor,   # (B, T, teacher_dim) — frozen CLIP features
    ) -> torch.Tensor:
        """
        L_spatial: MSE between projected student and teacher features at same timestep.
        Encourages student to encode similar visual semantics as CLIP.
        """
        B, T, _ = student_feats.shape
        s = self.spatial_proj(student_feats.reshape(B*T, -1))    # (B*T, teacher_dim)
        t = F.normalize(teacher_feats.reshape(B*T, -1), dim=-1)  # (B*T, teacher_dim)
        return F.mse_loss(s, t)

    def temporal_shift_loss(
        self,
        student_feats: torch.Tensor,   # (B, T, student_dim)
        teacher_feats: torch.Tensor,   # (B, T, teacher_dim) — frozen CLIP features
    ) -> torch.Tensor:
        """
        L_temporal: student(t) should predict teacher(t+1).
        This is the key innovation — teaches the student to "see the future".
        """
        if student_feats.shape[1] < 2:
            return student_feats.new_zeros(1).squeeze()

        B, T, _ = student_feats.shape

        # Student features at t=0..T-2
        s_t = student_feats[:, :-1, :].reshape(B*(T-1), -1)  # (B*(T-1), s_dim)
        # Teacher features at t=1..T-1 (shifted by 1)
        t_t1 = F.normalize(
            teacher_feats[:, 1:, :].reshape(B*(T-1), -1), dim=-1)  # (B*(T-1), t_dim)

        # Apply temporal transform then project
        s_transformed = self.temporal_proj(s_t)          # (B*(T-1), s_dim)
        s_projected   = self.temporal_head(s_transformed) # (B*(T-1), t_dim)

        return F.mse_loss(s_projected, t_t1)

    def contrastive_loss(
        self,
        concept_acts:   torch.Tensor,   # (B, T, num_concepts)
        text_embeddings: torch.Tensor,  # (num_concepts, teacher_dim) — frozen CLIP text
    ) -> torch.Tensor:
        """
        L_contrast: align concept activation patterns with CLIP text embeddings.
        This directly grounds concept bottleneck in CLIP's semantic space.
        Uses a CLIP-style contrastive objective.
        """
        B, T, C = concept_acts.shape

        # Mean pooling over time for sequence-level concept representation
        c_mean = concept_acts.mean(dim=1)  # (B, C)

        # Project concept activations to CLIP text space
        v = self.concept_proj(c_mean)       # (B, teacher_dim), L2-normalized

        # Text embeddings (frozen CLIP)
        z = F.normalize(text_embeddings, dim=-1)  # (C, teacher_dim)

        # Compute similarity: each sample's concept pattern vs all concept texts
        # We use top-k concept indices as positive pairs
        tau = torch.clamp(self.temperature, min=0.01, max=0.5)

        # Get top-k active concepts per sample as positives
        k = min(10, C)
        topk_indices = concept_acts.mean(dim=1).topk(k, dim=1).indices  # (B, k)

        # Compute video-text similarity
        sim = torch.mm(v, z.T) / tau  # (B, C)

        # Contrastive loss: each sample's top concepts as positives
        loss = torch.zeros(1, device=v.device).squeeze()
        for b in range(B):
            pos_mask = torch.zeros(C, device=v.device)
            pos_mask[topk_indices[b]] = 1.0 / k
            log_probs = F.log_softmax(sim[b], dim=0)
            loss = loss - (pos_mask * log_probs).sum()
        return loss / B

    def forward(
        self,
        student_hidden_seq: torch.Tensor,    # (B, T, student_dim)
        teacher_feat_seq:   torch.Tensor,    # (B, T, teacher_dim) frozen CLIP
        concept_acts_seq:   torch.Tensor,    # (B, T, num_concepts)
        text_embeddings:    torch.Tensor,    # (num_concepts, teacher_dim) frozen CLIP text
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute all distillation losses.
        Returns total distillation loss and individual loss dict.
        """
        l_spatial  = self.spatial_loss(student_hidden_seq, teacher_feat_seq)
        l_temporal = self.temporal_shift_loss(student_hidden_seq, teacher_feat_seq)
        l_contrast = self.contrastive_loss(concept_acts_seq, text_embeddings)

        total = (self.lambda_spatial  * l_spatial
               + self.lambda_temporal * l_temporal
               + self.lambda_contrast * l_contrast)

        return total, {
            'distill_spatial':  l_spatial,
            'distill_temporal': l_temporal,
            'distill_contrast': l_contrast,
        }


class CLIPTeacher(nn.Module):
    """
    Frozen CLIP image encoder as teacher.
    Extracts per-frame features from raw video frames.

    For our VGG-16 feature pipeline, we use a lightweight adapter
    to convert VGG-16 features to CLIP-compatible representations.
    This avoids running CLIP on raw pixels (too slow for training).
    """
    def __init__(
        self,
        vgg_dim:     int = 4096,   # VGG-16 feature dimension
        teacher_dim: int = 512,    # CLIP ViT-B/16 output dimension
        freeze:      bool = True,
    ):
        super().__init__()
        # Adapter: VGG features → CLIP-like semantic space
        # Trained jointly but constrained by CLIP text embeddings
        self.adapter = nn.Sequential(
            nn.Linear(vgg_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, teacher_dim),
        )
        if freeze:
            # Freeze adapter initially, only unfreeze after warmup
            for p in self.adapter.parameters():
                p.requires_grad = False

    def unfreeze(self):
        for p in self.adapter.parameters():
            p.requires_grad = True

    def forward(self, vgg_feats: torch.Tensor) -> torch.Tensor:
        """Convert VGG features to teacher-space features."""
        return F.normalize(self.adapter(vgg_feats), dim=-1)
