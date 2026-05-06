# -*- coding: utf-8 -*-
"""
Actor-Critic Module for CG-CRASH v4
====================================
灵感来源：AAAI 2025 "Diffusion + Actor-Critic for Accident Anticipation"

核心创新：Concept-Aware Actor-Critic
  - State: GRU 隐状态 + Concept 激活向量（可解释的 state 表示）
  - Actor: 决定"何时报警"（policy network）
  - Critic: 评估当前 state 的长期价值（value network）
  - Reward: concept 越早激活、越准确，奖励越高

双层可解释性：
  - WHY: Concept Bottleneck 解释"为什么危险"
  - WHEN: Actor 权重解释"为什么现在报警"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConceptAwareActorCritic(nn.Module):
    """
    Concept-Aware Actor-Critic Module.

    State = [h_t || c_t] where:
      h_t: GRU hidden state (h_dim)
      c_t: concept activations (num_concepts) — interpretable risk signal

    Actor outputs a scalar "alert confidence" used to gate the final prediction.
    Critic outputs a value estimate for the current state.

    Args:
        h_dim:        GRU hidden dimension
        num_concepts: number of CBM concepts
        ac_dim:       Actor-Critic internal dimension
        gamma:        discount factor for returns
        entropy_coef: entropy regularization coefficient
    """
    def __init__(
        self,
        h_dim:        int   = 256,
        num_concepts: int   = 837,
        ac_dim:       int   = 128,
        gamma:        float = 0.95,
        entropy_coef: float = 0.01,
    ):
        super().__init__()
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.h_dim        = h_dim
        self.num_concepts = num_concepts

        state_dim = h_dim + num_concepts

        # Actor: policy network → P(alert | state)
        # Outputs logits for binary action: {0: wait, 1: alert}
        self.actor = nn.Sequential(
            nn.Linear(state_dim, ac_dim),
            nn.LayerNorm(ac_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ac_dim, ac_dim // 2),
            nn.ReLU(),
            nn.Linear(ac_dim // 2, 2),  # binary: wait or alert
        )

        # Critic: value network → V(state)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, ac_dim),
            nn.LayerNorm(ac_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ac_dim, ac_dim // 2),
            nn.ReLU(),
            nn.Linear(ac_dim // 2, 1),  # scalar value
        )

        # Concept-to-alert projection: direct interpretable path
        # High-risk concepts → alert signal
        self.concept_alert_proj = nn.Linear(num_concepts, 1, bias=False)

        # Learned time-weight for anticipation: earlier alert = higher reward
        # This is the key to optimizing mTTA
        self.time_weight_fc = nn.Linear(h_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_state(self, h_t: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """Concatenate GRU hidden state with concept activations."""
        return torch.cat([h_t, c_t], dim=-1)  # (B, h_dim + num_concepts)

    def forward(
        self,
        h_t:    torch.Tensor,   # (B, h_dim) — GRU hidden state
        c_t:    torch.Tensor,   # (B, C) — concept activations
        action: Optional[torch.Tensor] = None,  # (B,) ground truth action if training
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: (B, 2) — policy logits
            value:         (B, 1) — state value estimate
            time_weight:   (B, 1) — temporal anticipation weight
        """
        state = self.get_state(h_t, c_t)  # (B, h_dim + C)
        action_logits = self.actor(state)
        value         = self.critic(state)
        time_weight   = torch.sigmoid(self.time_weight_fc(h_t))  # (B, 1)
        return action_logits, value, time_weight

    def compute_concept_aware_reward(
        self,
        c_t:       torch.Tensor,   # (B, C) concept activations at time t
        pred_prob: torch.Tensor,   # (B,) predicted accident probability
        labels:    torch.Tensor,   # (B,) ground truth label (0 or 1)
        t:         int,            # current frame index
        toa:       torch.Tensor,   # (B,) time-of-accident frame index
        fps:       float = 20.0,
        reward_scale: float = 5.0,
        penalty_scale: float = 0.5,
    ) -> torch.Tensor:
        """
        Concept-Aware Reward Design (AAAI25 style with concept bonus):

        For positive samples (accident):
          - Correct early alert: reward = exp(-(toa-t)/fps) * (1 + concept_bonus)
            Concept bonus: high activation of relevant concepts → extra reward
          - Wrong/late alert: penalty = -penalty_scale

        For negative samples (non-accident):
          - Correct no-alert: small positive reward
          - False alarm: penalty

        Returns reward: (B,)
        """
        B = c_t.shape[0]
        device = c_t.device

        # Concept bonus: sum of concept activations weighted by alert projection
        # High activation of risk-relevant concepts → bonus
        concept_bonus = torch.sigmoid(self.concept_alert_proj(c_t)).squeeze(1)  # (B,)

        # Time-to-accident reward: EARLIER prediction → HIGHER reward
        # time_diff = seconds remaining before accident (larger = earlier alert)
        time_diff = (toa.float() - t - 1).clamp(min=0.0) / fps  # seconds remaining
        # Linear reward proportional to time remaining (earlier = better)
        # Normalized by total_seconds to keep in [0, 1] range
        total_seconds = toa.float().mean() / fps + 1e-6
        early_reward = reward_scale * (time_diff / (total_seconds + 1e-6))  # (B,)

        # Combine with concept bonus
        pos_reward = early_reward * (1.0 + 0.5 * concept_bonus)  # (B,)

        # Assign rewards based on label and prediction
        is_positive = (labels > 0.5).float()        # (B,)
        is_correct  = ((pred_prob > 0.5).float() == labels.float()).float()  # (B,)

        reward = torch.where(
            is_positive.bool(),
            torch.where(is_correct.bool(), pos_reward, -penalty_scale * torch.ones(B, device=device)),
            torch.where(is_correct.bool(), 0.1 * torch.ones(B, device=device),
                        -penalty_scale * torch.ones(B, device=device))
        )

        # Normalize reward per batch for stability
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        return reward

    def compute_actor_critic_loss(
        self,
        action_logits_seq: List[torch.Tensor],  # T x (B, 2)
        values_seq:        List[torch.Tensor],  # T x (B, 1)
        rewards_seq:       List[torch.Tensor],  # T x (B,)
        time_weights_seq:  List[torch.Tensor],  # T x (B, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Actor-Critic losses over a full sequence.

        Returns:
            policy_loss: actor loss (negative expected reward)
            value_loss:  critic MSE loss
            entropy_loss: entropy regularization (exploration bonus)
        """
        T = len(action_logits_seq)
        B = action_logits_seq[0].shape[0]
        device = action_logits_seq[0].device

        # Compute discounted returns G_t = r_t + gamma*r_{t+1} + ...
        returns = []
        G = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            G = rewards_seq[t] + self.gamma * G
            returns.insert(0, G.clone())

        # Normalize returns for stability
        returns_tensor = torch.stack(returns, dim=0)  # (T, B)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / \
                         (returns_tensor.std() + 1e-8)

        policy_loss  = torch.zeros(1, device=device)
        value_loss   = torch.zeros(1, device=device)
        entropy_loss = torch.zeros(1, device=device)

        for t in range(T):
            logits = action_logits_seq[t]           # (B, 2)
            value  = values_seq[t].squeeze(1)       # (B,)
            G_t    = returns_tensor[t]              # (B,)
            tw     = time_weights_seq[t].squeeze(1) # (B,) time importance weight

            # Advantage: how much better than expected
            advantage = (G_t - value.detach())      # (B,)

            # Policy gradient loss (weighted by time importance)
            log_probs = F.log_softmax(logits, dim=-1)  # (B, 2)
            # Policy gradient: use sampled action log-prob (not always action=1)
            # Sample action from current policy
            probs = F.softmax(logits, dim=-1)  # (B, 2)
            actions = probs.argmax(dim=-1)      # (B,) greedy action for stability
            log_probs = F.log_softmax(logits, dim=-1)  # (B, 2)
            action_log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
            policy_loss = policy_loss - (tw * action_log_prob * advantage).mean()

            # Value loss: MSE between value estimate and actual return
            value_loss = value_loss + F.mse_loss(value, G_t.detach())

            # Entropy regularization: encourage exploration
            entropy = -(probs * log_probs).sum(dim=-1).mean()  # scalar
            entropy_loss = entropy_loss - entropy  # minimize negative entropy

        policy_loss  = policy_loss.clamp(-10.0, 10.0) / T
        value_loss   = value_loss   / T
        entropy_loss = entropy_loss / T

        return policy_loss, value_loss, self.entropy_coef * entropy_loss


class AnticipationLossWithAC(nn.Module):
    """
    Combined loss: Anticipation (supervised) + Actor-Critic (RL).

    L_total = L_anticipation + alpha * L_policy + beta * L_value + gamma * L_entropy

    This is the key loss function for CG-CRASH v4.
    """
    def __init__(
        self,
        alpha: float = 0.5,   # policy loss weight
        beta:  float = 0.5,   # value loss weight
        fps:   float = 20.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.fps   = fps

    def forward(
        self,
        base_loss:    torch.Tensor,
        policy_loss:  torch.Tensor,
        value_loss:   torch.Tensor,
        entropy_loss: torch.Tensor,
    ) -> torch.Tensor:
        total = (base_loss
                 + self.alpha * policy_loss
                 + self.beta  * value_loss
                 + entropy_loss)
        return total
