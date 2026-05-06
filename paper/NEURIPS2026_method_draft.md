# CG-CRASH v4: Concept-Guided Actor-Critic for Interpretable Accident Anticipation
## NeurIPS 2026 Draft — Method Section

---

## 3. Method

### 3.1 Problem Formulation

We formulate traffic accident anticipation as a sequential risk forecasting problem
with dual objectives: **(1) accuracy** — correctly identify accident-prone sequences,
and **(2) timeliness** — issue warnings as early as possible.

Given a dashcam video $\mathbf{V} = \{V_t\}_{t=1}^T$, a model $f_\theta$
predicts frame-wise accident probabilities $p_t = f_\theta(V_{1:t})$.
Time-to-Accident (TTA) is defined as $\Delta t = \tau - t^*$,
where $\tau$ is the ground-truth accident frame and
$t^* = \min\{t : p_t \geq \theta\}$ is the first frame exceeding threshold $\theta$.

**Key challenge**: existing methods treat accuracy and timeliness as separate objectives,
optimizing only classification loss. We propose a unified framework that jointly
optimizes *what* to detect (via concept bottleneck) and *when* to alert (via actor-critic).

---

### 3.2 Overall Architecture

CG-CRASH v4 consists of four tightly-coupled components:

1. **Concept Bottleneck Module (CBM)** — extracts interpretable concept activations
2. **Concept-Guided Temporal Attention (CGTA)** — attends over history using concept dynamics
3. **Concept Risk Score (CRS)** — learnable per-concept risk weights
4. **Concept-Aware Actor-Critic (CAAC)** — optimizes *when* to alert using RL

This design achieves **dual-layer interpretability**:
- **WHY** layer: CBM explains which visual concepts triggered the risk assessment
- **WHEN** layer: Actor-Critic explains why *this moment* was chosen for the alert

---

### 3.3 Concept Bottleneck Module

The CBM maps image embeddings $\mathbf{h}_t$ to $K$ concept activations:
$$\mathbf{c}_t = \text{ReLU}(W_c \mathbf{h}_t + b_c) \in \mathbb{R}^K$$

Concepts are grounded in natural language via CLIP alignment loss:
$$\mathcal{L}_{\text{align}} = \frac{1}{K}\sum_{k=1}^K (1 - \langle \hat{w}_k, \hat{e}_k \rangle)$$
where $\hat{w}_k$ is the normalized concept projection weight and
$\hat{e}_k$ is the CLIP text embedding of the $k$-th concept name.

---

### 3.4 Concept-Guided Temporal Attention (CGTA)

Conventional temporal attention uses hidden states as queries.
We instead use **concept activation deltas** $\Delta\mathbf{c}_t = \mathbf{c}_t - \mathbf{c}_{t-1}$
as queries, directly linking interpretable concept changes to temporal focus:

$$\alpha_t = \text{softmax}\left(\frac{(W_Q \Delta\mathbf{c}_t) \mathbf{H}_{<t}^\top}{\sqrt{d}}\right)$$
$$\mathbf{z}_t^{\text{CGTA}} = \alpha_t \mathbf{H}_{<t}$$

This means the model's temporal attention weights directly show
**which past timesteps became relevant when a concept changed**,
providing fine-grained temporal interpretability.

---

### 3.5 Concept Risk Score (CRS)

We learn per-concept risk weights $\mathbf{w}_r \in \mathbb{R}^K$:
$$r_t = \sum_{k=1}^K \sigma(w_{r,k}) \cdot c_{t,k}$$

These weights are directly interpretable: high $\sigma(w_{r,k})$ indicates
concept $k$ is safety-critical. We encourage sparsity via:
$$\mathcal{L}_{\text{sparse}} = \frac{1}{K}\sum_{k=1}^K \sigma(w_{r,k})$$

---

### 3.6 Concept-Aware Actor-Critic (CAAC)

We formulate alert timing as a sequential decision problem.
The **state** at time $t$ is:
$$\mathbf{s}_t = [\mathbf{h}_t \| \mathbf{c}_t] \in \mathbb{R}^{d + K}$$

Crucially, state $\mathbf{s}_t$ contains both GRU hidden states *and* concept activations,
making the Actor's decisions directly interpretable: we can examine which concepts
drove the alert decision at any moment.

**Actor** (policy network) outputs alert probability:
$$\pi_\theta(a_t | \mathbf{s}_t) = \text{softmax}(f_{\text{actor}}(\mathbf{s}_t))$$

**Critic** (value network) estimates long-horizon safety utility:
$$V_\phi(\mathbf{s}_t) = f_{\text{critic}}(\mathbf{s}_t)$$

**Concept-Aware Reward** incentivizes early, accurate, concept-grounded alerts:
$$r_t = R_{\text{scale}} \cdot e^{-0.5(\tau-t)/\text{fps}} \cdot (1 + 0.5 \cdot b_t^{\text{concept}})$$
where $b_t^{\text{concept}} = \sigma(W_b \mathbf{c}_t)$ is a concept-based bonus
that rewards alerts accompanied by high-confidence concept activations.

**Training objective** combines supervised and RL losses:
$$\mathcal{L} = \mathcal{L}_{\text{ant}} + \alpha \mathcal{L}_{\text{policy}} + \beta \mathcal{L}_{\text{value}} + \gamma \mathcal{L}_{\text{entropy}}$$

---

### 3.7 Temporally Shifted Distillation (TSD)

Inspired by ICLR 2026, we use a frozen CLIP image encoder as teacher.
The student learns to predict *future* CLIP features from *current* state:
$$\mathcal{L}_{\text{temporal}} = \|P_{\text{ST}}(\mathbf{h}_t) - \hat{f}_T(V_{t+1})\|_2^2$$

This "future-prediction" supervision enables the student to develop
anticipatory representations without video-pretrained teachers.

Contrastive loss aligns concept activations with CLIP text space,
grounding concepts in CLIP's rich semantic space:
$$\mathcal{L}_{\text{contrast}} = -\log\frac{\exp(\text{sim}(\mathbf{v}, \mathbf{z}^+)/\tau)}{\sum_j \exp(\text{sim}(\mathbf{v}, \mathbf{z}_j)/\tau)}$$

---

### 3.8 RWKV Temporal Module (Optional Backbone)

While GRU provides a strong sequential baseline, its hidden state grows linearly
with sequence length and struggles with very long-range dependencies.
We propose an optional **RWKV** backbone that replaces GRU with linear-complexity
recurrent attention:

$$\mathbf{h}_t = \sigma(W_r \mathbf{x}_t) \odot \sum_{s \leq t} e^{-(t-s)w} (W_k \mathbf{x}_s)^\top (W_v \mathbf{x}_s)$$

where $w \in \mathbb{R}^d$ are learnable time-decay weights.
RWKV retains $O(1)$ inference cost per step while capturing long-range
concept dynamics critical for accident anticipation.

A **random span masking** strategy (mask ratio $\rho=0.2$) during training
forces the model to reconstruct masked temporal segments from concept context,
improving robustness to partial occlusion and sensor dropout.

RWKV integrates directly with the CBM-AC framework:
the RWKV output $\mathbf{h}_t$ feeds both the CBM (for concept extraction)
and the CAAC state $\mathbf{s}_t = [\mathbf{h}_t \| \mathbf{c}_t]$.
This maintains full dual-layer interpretability while improving temporal modeling.

---

## 4. Experiments (Planned)

### 4.1 Datasets
- **DAD**: 1750 videos, 20fps, 100 frames (primary benchmark)
- **A3D**: 1201 videos, East Asian urban scenes
- **CRASH**: 4500 videos with missing data conditions

### 4.2 Metrics
- AP (Average Precision)
- mTTA (Mean Time-To-Accident)
- TTA@R80 (TTA at 80% recall)

### 4.3 Baselines
| Method | Venue | DAD AP | DAD mTTA |
|--------|-------|--------|----------|
| CRASH  | MM'24 | 65.3%  | 3.05s    |
| W3AL   | MM'24 | 69.2%  | 4.26s    |
| MASTTA | TCSVT'25 | 80.8% | 3.96s |
| CCAF-Net | NEURO'25 | 81.3% | 4.15s |
| LATTE  | IF'24  | 89.7%  | 4.49s    |
| AAAI25 | AAAI'25 | 91.2% | 4.59s   |
| **CG-CRASH v4 (Ours)** | NeurIPS'26 | **TBD** | **TBD** |

### 4.4 Interpretability Analysis (NeurIPS 卖点)
1. Concept activation timeline visualization
2. Actor decision boundary analysis
3. User study: human agreement with model alerts
4. Ablation: with/without CBM, with/without AC, with/without TSD

---

## 5. Key Claims (NeurIPS 审稿人关注点)

1. **首个双层可解释事故预测模型**: WHY (CBM) + WHEN (Actor-Critic)
2. **Concept-Actor 耦合**: concept state 直接驱动 actor decision，决策完全可追溯
3. **性能 SOTA on DAD** (target: >80% AP, 当前 v3_fixed_lr ep20 已达 65.5%，CBM 阶段刚开始)
4. **轻量化**: <30M params vs CCAF-Net (191M), LATTE (pretrained backbone)
5. **理论贡献**: Concept-Aware Reward 的形式化设计 + RWKV 线性复杂度时序建模
6. **多数据集验证**: DAD / A3D / CCD 三数据集，消融实验完整
7. **RWKV 时序建模**: O(1) 推理复杂度 + random span masking 提升鲁棒性
