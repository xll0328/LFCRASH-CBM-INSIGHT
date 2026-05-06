# -*- coding: utf-8 -*-
"""
LFCRASH-CBM v3: Concept-Gated CRASH (CG-CRASH)
================================================
Fixes from v2:
  - Numerically stable multi-task uncertainty weighting (log-sigma parameterization)
  - NaN-safe loss computation with clamping
  - Improved concept bottleneck with layer-norm and residual connection
"""
import os, sys, math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_CRASH_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'CRASH')
sys.path.insert(0, _CRASH_DIR)
from src.RSDlayerAttention import Encoder
from src.fft import SpectralGatingNetwork
from src.actor_critic import ConceptAwareActorCritic
from src.rwkv_module import RWKVBlock


class ObjectFocusAttention(nn.Module):
    def __init__(self, h_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.q1 = nn.Linear(h_dim, h_dim)
        self.q2 = nn.Linear(h_dim, h_dim)
        self.wk = nn.Linear(h_dim, h_dim)
        self.wv = nn.Linear(h_dim, h_dim)
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

    def forward(self, obj_embed, h):
        q1 = self.q1(h[0]).unsqueeze(1)
        q2 = self.q2(h[min(1, self.n_layers-1)]).unsqueeze(1)
        k  = self.wk(obj_embed); v = self.wv(obj_embed)
        scale = math.sqrt(v.size(-1))
        w = F.softmax(self.alpha1*torch.bmm(q1,k.transpose(1,2))/scale
                    + self.alpha2*torch.bmm(q2,k.transpose(1,2))/scale, dim=-1)
        return torch.bmm(w, v)


class TemporalFocusAttention(nn.Module):
    def __init__(self, h_dim, num_heads=4):
        super().__init__()
        assert h_dim % num_heads == 0
        self.h_dim = h_dim; self.num_heads = num_heads
        self.depth = h_dim // num_heads
        self.register_buffer('_pe', self._make_pe(h_dim, 200))
        self.Wq = nn.Linear(h_dim, h_dim, bias=False)
        self.Wk = nn.Linear(h_dim, h_dim, bias=False)
        self.Wv = nn.Linear(h_dim, h_dim, bias=False)
        self.out = nn.Linear(h_dim, h_dim)
        for l in (self.Wq, self.Wk, self.Wv, self.out):
            nn.init.kaiming_normal_(l.weight, a=math.sqrt(5))

    @staticmethod
    def _make_pe(d, max_len):
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        return pe.unsqueeze(0)

    def _split(self, x):
        B,T,D=x.shape
        return x.view(B,T,self.num_heads,self.depth).transpose(1,2)

    def forward(self, h_seq):
        B,T,D = h_seq.shape
        x = h_seq + self._pe[:,:T,:]
        Q,K,V = self._split(self.Wq(x)), self._split(self.Wk(x)), self._split(self.Wv(x))
        a = F.softmax(torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.depth), dim=-1)
        o = torch.matmul(a,V).transpose(1,2).contiguous().view(B,T,D)
        o = self.out(o)
        return torch.cat([o.mean(1), o.max(1)[0]], dim=1)


class AccidentPredictor(nn.Module):
    def __init__(self, in_dim, out_dim=2, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim,64), nn.ReLU(), nn.Linear(64,out_dim))
    def forward(self, x): return self.net(x)


class GRUNet(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(in_dim, h_dim, n_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(h_dim, 64)
        self.fc2  = nn.Linear(64, out_dim)
        self.relu = nn.ReLU()
    def forward(self, x, h):
        o, h = self.gru(x, h)
        o = self.relu(self.fc1(self.drop(o[:,-1])))
        return self.fc2(o), h


class ConceptBottleneck(nn.Module):
    def __init__(self, h_dim, num_concepts, use_relu=True, dropout=0.1):
        super().__init__()
        self.use_relu = use_relu
        self.num_concepts = num_concepts
        self.concept_proj = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, num_concepts),
        )
        self.concept_decode = nn.Sequential(
            nn.Linear(num_concepts, h_dim), nn.ReLU(),
        )
        self.ln = nn.LayerNorm(h_dim)
        self.register_buffer('concept_embeddings', None)

    def encode(self, img_embed):
        acts = self.concept_proj(img_embed)
        return F.relu(acts) if self.use_relu else acts

    def decode(self, acts):
        return self.concept_decode(acts)

    def forward(self, img_embed):
        acts  = self.encode(img_embed)
        embed = self.ln(self.decode(acts))
        return acts, embed

    def align_loss(self):
        W = self.concept_proj[-1].weight
        if self.concept_embeddings is None:
            return W.new_zeros(1).squeeze()
        W_n   = F.normalize(W, dim=1)
        C_emb = self.concept_embeddings.to(W.device)
        if C_emb.shape[1] != W.shape[1]:
            if not hasattr(self, '_clip_proj'):
                self._clip_proj = nn.Linear(C_emb.shape[1], W.shape[1], bias=False).to(W.device)
                nn.init.xavier_uniform_(self._clip_proj.weight)
            C_emb = F.normalize(self._clip_proj(C_emb), dim=1)
        else:
            C_emb = F.normalize(C_emb, dim=1)
        return (1.0 - (W_n * C_emb).sum(1)).mean()

    def sparse_loss(self, acts): return acts.abs().mean()

    def recon_loss(self, embed, img_embed):
        return F.mse_loss(embed, img_embed.detach())


class LFCRASH_CBM_GRU(nn.Module):
    _FFT_DIM: int = 512

    def __init__(
        self,
        x_dim:         int   = 4096,
        h_dim:         int   = 256,
        z_dim:         int   = 128,
        n_layers:      int   = 2,
        n_obj:         int   = 19,
        n_frames:      int   = 100,
        fps:           float = 20.0,
        with_saa:      bool  = True,
        num_concepts:  int   = 837,
        concept_file:  str   = None,
        clip_model:    str   = 'ViT-B/16',
        lambda_align:  float = 1e-4,
        lambda_sparse: float = 1e-3,
        lambda_recon:  float = 1e-2,
        use_cbm:       bool  = True,
        device:        str   = 'cuda',
        legacy:        bool  = False,  # load old checkpoints without CGTA/CRS
        use_rwkv:      bool  = False,  # use RWKV temporal module instead of GRU
    ):
        super().__init__()
        self.h_dim        = h_dim
        self.n_layers     = n_layers
        self.fps          = fps
        self.with_saa     = with_saa
        self.use_rwkv     = use_rwkv
        self._rwkv_state  = None
        self.lambda_align  = lambda_align
        self.lambda_sparse = lambda_sparse
        self.lambda_recon  = lambda_recon
        self.use_cbm      = use_cbm
        self.enable_cgta  = True
        self.enable_crs   = True
        self.ac_use_concepts = True
        D = self._FFT_DIM

        # Kendall et al. 2018: log(sigma^2) parameterization for stable multi-task weighting
        self.log_sigma_sq_1 = nn.Parameter(torch.zeros(1))
        self.log_sigma_sq_2 = nn.Parameter(torch.zeros(1))

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim * 2), nn.ReLU(),
            nn.Linear(h_dim * 2, h_dim), nn.ReLU(),
        )

        self.cbm = ConceptBottleneck(h_dim, num_concepts)
        self.ofa = ObjectFocusAttention(h_dim, n_layers)

        self.fft_in    = nn.Linear(h_dim, D) if h_dim != D else None
        self.fft_block = SpectralGatingNetwork(dim=3)
        self.fft_out   = nn.Linear(1, h_dim)

        # ── Concept-Guided Temporal Attention (CGTA) ──────────────────────
        # Uses concept activation dynamics (Δc_t) to attend over GRU history.
        # This ties interpretability directly to prediction: the attention weights
        # show WHICH timesteps triggered the alert via WHICH concept changes.
        self.cgta_q = nn.Linear(num_concepts, h_dim, bias=False)  # concept delta → query
        self.cgta_k = nn.Linear(h_dim, h_dim, bias=False)          # hidden → key
        self.cgta_v = nn.Linear(h_dim, h_dim, bias=False)          # hidden → value
        self.cgta_gate = nn.Parameter(torch.tensor(0.1))           # learnable gate

        # ── Concept Risk Score (CRS) ──────────────────────────────────────────
        # Learnable per-concept risk weights → direct interpretable risk signal
        self.concept_risk_w = nn.Parameter(torch.zeros(num_concepts))
        self.crs_proj = nn.Linear(1, h_dim)  # risk scalar → feature

        gru_in = 3 * h_dim + h_dim  # add CGTA context (v3+)
        self._legacy = legacy
        if legacy:
            gru_in = 3 * h_dim  # old checkpoint format
        if use_rwkv:
            # RWKV-based temporal module: fewer params, better long-range modeling
            self.gru = RWKVBlock(in_dim=gru_in, h_dim=h_dim, out_dim=2,
                                  n_layers=n_layers, mask_ratio=0.2)
        else:
            self.gru = GRUNet(gru_in, h_dim, 2, n_layers)

        self.rsd     = Encoder(v_hidden_size=D, v_num_attention_heads=8)
        self.rsd_in  = nn.Linear(h_dim, D) if h_dim != D else None
        self.rsd_out = nn.Linear(D, h_dim) if h_dim != D else None

        if with_saa:
            self.tfa           = TemporalFocusAttention(h_dim)
            self.aux_predictor = AccidentPredictor(2 * h_dim)

        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none')

        # ── Actor-Critic Module (v4) ──────────────────────────────────────
        # Concept-Aware Actor-Critic for optimizing WHEN to alert (mTTA)
        # State = [h_t || c_t] → interpretable decision making
        self.use_ac = True
        self.ac_module = ConceptAwareActorCritic(
            h_dim=h_dim,
            num_concepts=num_concepts,
            ac_dim=128,
            gamma=0.95,
            entropy_coef=0.01,
        )
        self.lambda_ac_policy  = 0.5
        self.lambda_ac_value   = 0.5

        if concept_file is not None:
            self._encode_concepts(concept_file, clip_model, device)

    def _encode_concepts(self, concept_file, clip_model_name, device):
        import hashlib
        import clip as _clip
        concept_path = os.path.abspath(concept_file)
        with open(concept_path) as f:
            concepts = [l.strip() for l in f if l.strip()]

        cache_root = os.path.join(os.path.dirname(__file__), '..', 'output', 'concept_cache')
        os.makedirs(cache_root, exist_ok=True)
        cache_key_src = clip_model_name + '||' + concept_path + '||' + str(len(concepts))
        cache_key = hashlib.md5(cache_key_src.encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_root, f'{cache_key}.pt')

        if os.path.exists(cache_path):
            print(f'[CG-CRASH] Loading cached concept embeddings: {cache_path}', flush=True)
            embs = torch.load(cache_path, map_location='cpu')
            self.cbm.concept_embeddings = embs.float().cpu()
            print(f'[CG-CRASH] {len(concepts)} cached concepts, shape {self.cbm.concept_embeddings.shape}', flush=True)
            return

        print(f'[CG-CRASH] Encoding concepts via CLIP {clip_model_name}...', flush=True)
        clip_model, _ = _clip.load(clip_model_name, device=device)
        clip_model.eval()
        with torch.no_grad():
            tokens = _clip.tokenize(concepts).to(device)
            embs   = clip_model.encode_text(tokens).float()
            embs   = F.normalize(embs, dim=1)
        embs_cpu = embs.cpu()
        torch.save(embs_cpu, cache_path)
        self.cbm.concept_embeddings = embs_cpu
        print(f'[CG-CRASH] Saved concept embedding cache: {cache_path}', flush=True)
        print(f'[CG-CRASH] {len(concepts)} concepts, shape {embs.shape}', flush=True)

    def _apply_rsd(self, h_history: List[torch.Tensor]) -> torch.Tensor:
        seq = len(h_history)
        B   = h_history[0].size(1)
        D   = self._FFT_DIM

        l0 = torch.stack([h[0] for h in h_history], dim=1)
        l1 = torch.stack([h[min(1, self.n_layers-1)] for h in h_history], dim=1)

        if self.rsd_in is not None:
            l0 = self.rsd_in(l0.reshape(-1, self.h_dim)).view(B, seq, D)
            l1 = self.rsd_in(l1.reshape(-1, self.h_dim)).view(B, seq, D)

        h_stack = torch.stack([l0, l1], dim=1)
        enc_out = self.rsd(h_stack)
        agg0 = enc_out[0, -1]
        agg1 = enc_out[1, -1]
        enc  = torch.stack([
            agg0.unsqueeze(0).expand(B, D).contiguous(),
            agg1.unsqueeze(0).expand(B, D).contiguous(),
        ], dim=0)

        if self.rsd_out is not None:
            enc = self.rsd_out(enc.reshape(2*B, D)).view(2, B, self.h_dim)

        h_out_dim = enc.size(-1)
        if self.n_layers > 2:
            pad = enc[-1:].expand(self.n_layers-2, B, h_out_dim)
            enc = torch.cat([enc, pad], dim=0)
        elif self.n_layers == 1:
            enc = enc[:1]
        return enc.contiguous()

    def _exp_loss(self, pred, target, t, toa):
        """Exponential anticipation loss with NaN-safe clamping."""
        cls = target[:, 1].long()
        time_diff = (toa.to(pred.dtype) - t - 1) / self.fps
        penalty = -0.5 * torch.clamp(time_diff, min=0.0)
        ce = self.ce_loss_fn(pred, cls)
        ce = torch.clamp(ce, max=50.0)
        pos = torch.exp(penalty) * ce
        neg = ce
        loss = torch.mean(
            target[:, 1].to(pred.dtype) * pos +
            target[:, 0].to(pred.dtype) * neg)
        return loss

    def forward(self, x, y, toa, hidden_in=None):
        B, T, Np1, _ = x.shape
        device = x.device

        h = (torch.zeros(self.n_layers, B, self.h_dim, device=device)
             if hidden_in is None else hidden_in)
        if self.use_rwkv:
            # Each batch item is an independent sequence; do not leak RWKV state.
            self._rwkv_state = None

        losses = {k: x.new_zeros(1).squeeze() for k in
                  ('total_loss','ce_loss','aux_loss',
                   'align_loss','sparse_loss','recon_loss',
                   'ac_policy_loss','ac_value_loss','ac_entropy_loss')}

        all_outputs: List[torch.Tensor] = []
        all_hidden:  List[torch.Tensor] = []
        h_history:   List[torch.Tensor] = []
        concept_acts_list: List[torch.Tensor] = []
        img_embed_list:    List[torch.Tensor] = []
        concept_embed_list: List[torch.Tensor] = []
        # Actor-Critic sequence buffers
        ac_action_logits_seq: List[torch.Tensor] = []
        ac_values_seq:        List[torch.Tensor] = []
        ac_rewards_seq:       List[torch.Tensor] = []
        ac_time_weights_seq:  List[torch.Tensor] = []

        RSD_WINDOW = 10
        prev_c_act = None  # for CGTA delta computation

        for t in range(T):
            frame = x[:, t]

            feats   = self.phi_x(frame)
            img_emb = feats[:, 0]
            obj_emb = feats[:, 1:]

            if self.use_cbm:
                c_act, c_embed = self.cbm(img_emb)
            else:
                c_act  = img_emb.new_zeros(B, self.cbm.num_concepts)
                c_embed = img_emb

            concept_acts_list.append(c_act)
            img_embed_list.append(img_emb)
            concept_embed_list.append(c_embed if self.use_cbm else None)

            obj_ctx = self.ofa(obj_emb, h)
            obj_vec = obj_ctx.squeeze(1)

            fft_in = self.fft_in(img_emb) if self.fft_in is not None else img_emb
            fft_out = self.fft_block(fft_in.unsqueeze(-1))
            fft_vec = self.fft_out(fft_out.mean(dim=1))

            # ── CGTA: Concept-Guided Temporal Attention ───────────────────
            # Compute concept activation delta (rate of change)
            if prev_c_act is not None:
                delta_c = c_act - prev_c_act          # (B, C) concept dynamics
            else:
                delta_c = torch.zeros_like(c_act)
            prev_c_act = c_act.detach()

            if not getattr(self, '_legacy', False):
                if len(all_hidden) > 0 and self.enable_cgta:
                    # Attend over past hidden states using concept delta as query
                    h_stack = torch.stack(all_hidden, dim=1)           # (B, t, h_dim)
                    cgta_q  = self.cgta_q(delta_c).unsqueeze(1)        # (B, 1, h_dim)
                    cgta_k  = self.cgta_k(h_stack)                     # (B, t, h_dim)
                    cgta_v  = self.cgta_v(h_stack)                     # (B, t, h_dim)
                    scale   = math.sqrt(self.h_dim)
                    attn_w  = F.softmax(torch.bmm(cgta_q, cgta_k.transpose(1,2)) / scale, dim=-1)
                    cgta_ctx = torch.bmm(attn_w, cgta_v).squeeze(1)    # (B, h_dim)
                    cgta_ctx = torch.tanh(self.cgta_gate) * cgta_ctx   # gated
                else:
                    cgta_ctx = img_emb.new_zeros(B, self.h_dim)

                # ── CRS: Concept Risk Score ───────────────────────────────────
                # Learnable weighted sum of concept activations → risk signal
                if self.enable_crs:
                    risk_w  = torch.sigmoid(self.concept_risk_w)           # (C,) in [0,1]
                    risk_score = (c_act * risk_w).sum(dim=1, keepdim=True) # (B, 1)
                    risk_feat   = self.crs_proj(risk_score)                # (B, h_dim)
                else:
                    risk_feat = img_emb.new_zeros(B, self.h_dim)

                gru_in = torch.cat([obj_vec, c_embed, fft_vec, cgta_ctx + risk_feat], dim=1).unsqueeze(1)
            else:
                # Legacy mode: old checkpoint without CGTA/CRS
                gru_in = torch.cat([obj_vec, c_embed, fft_vec], dim=1).unsqueeze(1)

            if self.use_rwkv:
                out_t, h_last = self.gru.forward_step(gru_in.squeeze(1), state=self._rwkv_state)
                self._rwkv_state = h_last.unsqueeze(0) if h_last.dim() == 2 else h_last
                # For RWKV, maintain a pseudo GRU-compatible hidden state
                h_last_expand = h_last.unsqueeze(0).expand(self.n_layers, -1, -1).contiguous()
                h = h_last_expand
            else:
                out_t, h = self.gru(gru_in, h)
            all_outputs.append(out_t)
            all_hidden.append(h[-1])
            h_history.append(h.detach())

            if len(h_history) >= RSD_WINDOW and (t+1) % RSD_WINDOW == 0:
                h = self._apply_rsd(h_history[-RSD_WINDOW:])

            if y is not None and toa is not None:
                losses['ce_loss'] = losses['ce_loss'] + self._exp_loss(out_t, y, t, toa) / T

            # ── Actor-Critic: collect per-step data ───────────────────────
            if self.use_ac and y is not None:
                ac_c_act = c_act if self.ac_use_concepts else torch.zeros_like(c_act)
                ac_logits, ac_value, ac_tw = self.ac_module(h[-1], ac_c_act)
                ac_action_logits_seq.append(ac_logits)
                ac_values_seq.append(ac_value)
                ac_time_weights_seq.append(ac_tw)
                # Compute concept-aware reward
                pred_prob = torch.softmax(out_t, dim=-1)[:, 1].detach()
                labels    = y[:, 1]
                reward    = self.ac_module.compute_concept_aware_reward(
                    c_t=c_act.detach(), pred_prob=pred_prob,
                    labels=labels, t=t, toa=toa, fps=self.fps,
                )
                ac_rewards_seq.append(reward)

        if self.with_saa and y is not None:
            h_seq    = torch.stack(all_hidden, dim=1)
            agg      = self.tfa(h_seq)
            aux_pred = self.aux_predictor(agg)
            aux_ce   = self.ce_loss_fn(aux_pred, y[:,1].long()).mean()
            aux_ce   = torch.clamp(aux_ce, max=50.0)
            losses['aux_loss'] = aux_ce

        if y is not None and self.use_cbm:
            c_all = torch.stack(concept_acts_list, dim=1)
            losses['align_loss']  = self.cbm.align_loss()
            losses['sparse_loss'] = self.cbm.sparse_loss(c_all)
            c_emb_all = torch.stack(concept_embed_list, dim=1)
            i_emb_all = torch.stack(img_embed_list,     dim=1)
            losses['recon_loss']  = self.cbm.recon_loss(
                c_emb_all.reshape(-1, self.h_dim),
                i_emb_all.reshape(-1, self.h_dim)
            )
            # CRS sparsity: encourage few high-risk concepts (interpretable)
            if not getattr(self, '_legacy', False) and self.enable_crs:
                losses['sparse_loss'] = losses['sparse_loss'] + \
                    self.lambda_sparse * torch.sigmoid(self.concept_risk_w).sum() / self.cbm.num_concepts

        # ── Actor-Critic loss over full sequence ─────────────────────────
        if self.use_ac and y is not None and len(ac_action_logits_seq) > 0:
            pol_loss, val_loss, ent_loss = self.ac_module.compute_actor_critic_loss(
                ac_action_logits_seq, ac_values_seq,
                ac_rewards_seq, ac_time_weights_seq,
            )
            losses['ac_policy_loss']  = torch.clamp(pol_loss,  -10.0, 10.0)
            losses['ac_value_loss']   = torch.clamp(val_loss,   0.0,  10.0)
            losses['ac_entropy_loss'] = torch.clamp(ent_loss,  -5.0,  5.0)

        if y is not None:
            # Stable multi-task weighting: L_total = L1/(2*sigma1^2) + L2/(2*sigma2^2) + log(sigma1*sigma2)
            # With s = log(sigma^2): L/(2*exp(s)) + s/2
            # Clamp log_sigma_sq to [-2, 2] to prevent:
            #   s1 << 0  => exp(-s1) >> 1  => CE loss explodes (observed: ce=41 on DAD)
            #   s1 >> 0  => total loss goes negative (observed: loss=-0.72 on no-CBM)
            s1 = self.log_sigma_sq_1.clamp(-2.0, 2.0)
            s2 = self.log_sigma_sq_2.clamp(-2.0, 2.0)
            weighted_ce  = losses['ce_loss']  * torch.exp(-s1) + 0.5 * s1
            weighted_aux = losses['aux_loss'] * torch.exp(-s2) + 0.5 * s2

            losses['total_loss'] = (
                weighted_ce
                + weighted_aux
                + self.lambda_align  * losses['align_loss']
                + self.lambda_sparse * losses['sparse_loss']
                + self.lambda_recon  * losses['recon_loss']
                + self.lambda_ac_policy * losses['ac_policy_loss']
                + self.lambda_ac_value  * losses['ac_value_loss']
                + losses['ac_entropy_loss']
            )

        return losses, all_outputs, all_hidden

    @torch.no_grad()
    def get_concept_activations(self, x):
        B, T, _, _ = x.shape
        acts = []
        for t in range(T):
            img_emb = self.phi_x(x[:,t,0,:])
            acts.append(self.cbm.encode(img_emb))
        return torch.stack(acts, dim=1)

    @torch.no_grad()
    def get_concept_risk_weights(self):
        """Return learned per-concept risk weights (CRS).
        High values = concepts the model considers high-risk.
        Directly interpretable: top-k concepts with highest weights
        are the model's learned safety-critical concepts.
        """
        return torch.sigmoid(self.concept_risk_w).cpu().numpy()

    @torch.no_grad()
    def get_cgta_attention(self, x):
        """Return CGTA attention weights for visualization.
        Shape: (T, T) — attn[t, s] = how much timestep t attended to past timestep s.
        Shows WHEN the model noticed concept changes that triggered alert.
        """
        B, T, _, _ = x.shape
        assert B == 1, 'get_cgta_attention expects batch size 1'
        all_hidden = []
        prev_c_act = None
        attn_maps = []
        h = torch.zeros(self.n_layers, B, self.h_dim, device=x.device)
        self._rwkv_state = None  # reset RWKV state at start of each sequence
        for t in range(T):
            img_emb = self.phi_x(x[:,t,0,:])
            if self.use_cbm:
                c_act, _ = self.cbm(img_emb)
            else:
                c_act = img_emb.new_zeros(B, self.cbm.num_concepts)
            delta_c = (c_act - prev_c_act) if prev_c_act is not None else torch.zeros_like(c_act)
            prev_c_act = c_act.detach()
            if len(all_hidden) > 0:
                h_stack = torch.stack(all_hidden, dim=1)
                cgta_q  = self.cgta_q(delta_c).unsqueeze(1)
                cgta_k  = self.cgta_k(h_stack)
                scale   = math.sqrt(self.h_dim)
                attn_w  = F.softmax(torch.bmm(cgta_q, cgta_k.transpose(1,2)) / scale, dim=-1)
                attn_maps.append(attn_w.squeeze().cpu().numpy())
            else:
                attn_maps.append(None)
            all_hidden.append(h[-1].detach())
        return attn_maps

    @torch.no_grad()
    def get_top_concepts(self, x, k=10):
        acts = self.get_concept_activations(x).mean(dim=1)
        scores, indices = acts.topk(k, dim=1)
        return indices, scores
