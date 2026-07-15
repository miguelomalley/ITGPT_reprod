import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence


# -----------------------------
# Utility: masks
# -----------------------------
def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def make_key_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    B = lengths.shape[0]
    if max_len is None:
        max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
    return arange >= lengths.unsqueeze(1)


class MotifSequenceConditioner(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.m_ln = nn.LayerNorm(d_model)
        self.x_ln = nn.LayerNorm(d_model)

        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tok_x: torch.Tensor,
        motif_seq: torch.Tensor,
        motif_activity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gamma, beta = self.net(self.m_ln(motif_seq)).chunk(2, dim=-1)
        update = gamma * self.x_ln(tok_x) + beta

        if motif_activity is not None:
            update = update * motif_activity

        return tok_x + self.scale * self.drop(update)


# -----------------------------
# Causal motif temporal contextualizer
# -----------------------------
class CausalMotifContextBlock(nn.Module):
    """
    Small pre-norm causal Transformer block over the motif evidence sequence.

    x: (B, T, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        y, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x


# -----------------------------
# Causal motif detector
# -----------------------------
class CausalMotifDetector(nn.Module):
    """
    Causal per-position motif detector.

    High-level flow
    ---------------
    1. Multi-scale causal token+aux windows -> phrase_agg: (B, T, hidden_dim)
    2. phrase_agg matched against motif_keys -> raw motif logits
    3. audio supplies weak identity prior and stronger activity prior
    4. top-k sparse motif assignment -> raw motif vector
    5. recent motif-frequency summaries + phrase projection + raw motif vector
       form a motif evidence sequence
    6. causal motif self-attention contextualizes the evidence over time
    7. context refines motif identity and activity
    8. final motif_seq: (B, T, d_model) conditions the decoder
    """

    def __init__(self, config):
        super().__init__()

        self.n_motifs = config.n_motifs
        self.d_model = config.d_model
        self.hidden_dim = config.motif_hidden_dim
        self.aux_dim = config.aux_dim
        self.window_sizes = list(config.motif_window_sizes)
        self.pad_id = config.num_classes

        phrase_dim = config.motif_phrase_dim
        dropout = config.dropout

        self.temperature = getattr(config, "motif_temperature", 0.15)
        self.top_k = getattr(config, "motif_top_k", 8)

        # Recent motif statistics windows.
        self.recent_windows = list(getattr(config, "motif_recent_windows", (8, 16, 32)))

        # -----------------------------
        # Local phrase encoders
        # -----------------------------
        self.phrase_emb = nn.Embedding(
            config.num_classes + 1,
            phrase_dim,
            padding_idx=config.num_classes,
        )

        self.aux_projs = nn.ModuleList([
            nn.Linear(win * config.aux_dim, phrase_dim)
            for win in self.window_sizes
        ])

        self.phrase_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(win * phrase_dim + phrase_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )
            for win in self.window_sizes
        ])

        self.scale_weights = nn.Parameter(torch.zeros(len(self.window_sizes)))

        # -----------------------------
        # Motif codebook
        # -----------------------------
        self.motif_keys = nn.Embedding(self.n_motifs, self.hidden_dim)
        nn.init.normal_(self.motif_keys.weight, std=0.02)

        self.motif_values = nn.Embedding(self.n_motifs, self.d_model)
        nn.init.normal_(self.motif_values.weight, std=0.02)

        # -----------------------------
        # Audio priors
        # -----------------------------
        self.audio_to_hidden = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.token_activity = nn.Linear(self.hidden_dim, 1)
        self.audio_activity = nn.Linear(self.d_model, 1)

        self.audio_identity_scale = nn.Parameter(torch.tensor(0.1))
        self.audio_activity_scale = nn.Parameter(torch.tensor(1.0))

        # -----------------------------
        # Evidence construction
        # -----------------------------
        self.phrase_to_model = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.d_model),
        )

        self.recent_proj = nn.Sequential(
            nn.LayerNorm(self.d_model * len(self.recent_windows)),
            nn.Linear(self.d_model * len(self.recent_windows), self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.motif_pos = nn.Embedding(config.max_len, self.d_model)

        # -----------------------------
        # Temporal motif contextualizer
        # -----------------------------
        n_context_layers = getattr(config, "motif_context_layers", 2)

        self.motif_context_blocks = nn.ModuleList([
            CausalMotifContextBlock(
                d_model=self.d_model,
                n_heads=config.motif_n_heads,
                dropout=dropout,
            )
            for _ in range(n_context_layers)
        ])

        # Context refinement heads.
        self.context_to_logits = nn.Linear(self.d_model, self.n_motifs)
        self.context_to_activity = nn.Linear(self.d_model, 1)

        # Start context refinement near no-op.
        nn.init.zeros_(self.context_to_logits.weight)
        nn.init.zeros_(self.context_to_logits.bias)
        nn.init.zeros_(self.context_to_activity.weight)
        nn.init.zeros_(self.context_to_activity.bias)

        # Important: no LayerNorm after activity scaling.
        # LayerNorm there would mostly erase the magnitude meaning of activity.
        self.out_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    # Alignment/window helpers
    # ------------------------------------------------------------------
    def _extract_windows(self, token_ids: torch.Tensor, win_len: int) -> torch.Tensor:
        """
        token_ids: (B, T)

        Returns causal windows ending at each t:
            (B, T, win_len)
        """
        B, T = token_ids.shape
        pad = token_ids.new_full((B, win_len - 1), self.pad_id)
        padded = torch.cat([pad, token_ids], dim=1)
        return padded.unfold(1, win_len, 1)

    def _align_aux(self, aux: torch.Tensor, T: int) -> torch.Tensor:
        L = aux.size(1)
        if L >= T:
            return aux[:, :T, :]
        return torch.cat([aux, aux[:, -1:, :].expand(-1, T - L, -1)], dim=1)
    
    def _shift_aux_for_token_history(self, aux: torch.Tensor, T: int) -> torch.Tensor:
        aux = self._align_aux(aux, T)
        first = aux[:, :1, :]
        return torch.cat([first, aux[:, :-1, :]], dim=1)

    def _align_audio(self, audio_emb: torch.Tensor, T: int) -> torch.Tensor:
        L = audio_emb.size(1)
        if L >= T:
            return audio_emb[:, :T, :]
        return torch.cat([audio_emb, audio_emb[:, -1:, :].expand(-1, T - L, -1)], dim=1)

    # ------------------------------------------------------------------
    # Local causal token+aux phrase encoding
    # ------------------------------------------------------------------
    def encode_multiscale_token_aux_windows(
        self,
        token_ids: torch.Tensor,
        aux_steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-scale causal phrase encoder.

        token_ids: (B, T)
        aux_steps: (B, T_aux, aux_dim)

        Returns:
            phrase_agg: (B, T, hidden_dim)
        """
        B, T = token_ids.shape
        aux = self._align_aux(aux_steps, T)

        phrase_per_scale = []

        for win_len, aux_proj, enc in zip(
            self.window_sizes,
            self.aux_projs,
            self.phrase_encoders,
        ):
            # Token causal window.
            windows = self._extract_windows(token_ids, win_len)       # (B, T, win)
            tok_feat = self.phrase_emb(windows).reshape(B, T, -1)     # (B, T, win * phrase_dim)

            # Aux causal window.
            aux_t = F.pad(aux.permute(0, 2, 1), (win_len - 1, 0))     # (B, aux_dim, T+win-1)
            aux_win = aux_t.unfold(2, win_len, 1).permute(0, 2, 3, 1) # (B, T, win, aux_dim)
            aux_win = aux_win.reshape(B, T, win_len * self.aux_dim)
            aux_feat = aux_proj(aux_win)                              # (B, T, phrase_dim)

            phrase = enc(torch.cat([tok_feat, aux_feat], dim=-1))     # (B, T, hidden_dim)
            phrase_per_scale.append(phrase)

        scale_w = torch.softmax(self.scale_weights, dim=0)
        phrase_agg = sum(w * p for w, p in zip(scale_w, phrase_per_scale))

        return phrase_agg

    # ------------------------------------------------------------------
    # Sparse motif assignment
    # ------------------------------------------------------------------
    def _topk_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sparse top-k softmax over motif codes.

        logits: (B, T, n_motifs)
        returns probs with only top_k nonzero entries per position.
        """
        k = min(self.top_k, logits.size(-1))
        topv, topi = torch.topk(logits, k=k, dim=-1)
        topp = torch.softmax(topv / self.temperature, dim=-1)

        probs = torch.zeros_like(logits)
        probs.scatter_(-1, topi, topp)
        return probs

    # ------------------------------------------------------------------
    # Recent motif-frequency summaries
    # ------------------------------------------------------------------
    @staticmethod
    def _causal_window_average(x: torch.Tensor, window: int) -> torch.Tensor:
        """
        Causal sliding-window average.

        x: (B, T, D)
        returns: (B, T, D)

        At position t, average over max(0, t-window+1) ... t.
        """
        B, T, D = x.shape
        csum = torch.cumsum(x, dim=1)
        zero = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        csum = torch.cat([zero, csum], dim=1)  # (B, T+1, D)

        end = torch.arange(1, T + 1, device=x.device)
        start = torch.clamp(end - window, min=0)

        summed = csum[:, end, :] - csum[:, start, :]
        denom = (end - start).float().view(1, T, 1).clamp_min(1.0)

        return summed / denom

    def compute_recent_motif_vectors(self, motif_probs: torch.Tensor) -> torch.Tensor:
        """
        motif_probs: (B, T, n_motifs), normally raw assignment probabilities.

        Returns:
            recent_cat: (B, T, d_model * len(recent_windows))
        """
        recent_vecs = []

        for w in self.recent_windows:
            recent_probs = self._causal_window_average(motif_probs, w)      # (B, T, n_motifs)
            recent_vec = recent_probs @ self.motif_values.weight            # (B, T, d_model)
            recent_vecs.append(recent_vec)

        return torch.cat(recent_vecs, dim=-1)

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------
    def _auxiliary_losses(
        self,
        phrase_agg_n: torch.Tensor,
        motif_probs: torch.Tensor,
        keys_n: torch.Tensor,
        activity: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Lightweight codebook regularization.

        phrase_agg_n: (B, T, hidden_dim), normalized
        motif_probs:  (B, T, n_motifs), assignment probabilities, preferably sum=1
        keys_n:       (n_motifs, hidden_dim), normalized
        activity:     optional (B, T, 1)

        Notes:
        - We use motif_probs for code assignment, not activity-scaled probabilities.
        - Activity can weight the VQ-like loss so inactive positions matter less.
        """
        B, T, H = phrase_agg_n.shape
        device = phrase_agg_n.device

        with torch.no_grad():
            hard_idx = motif_probs.argmax(dim=-1)  # (B, T)

        hard_keys = keys_n[hard_idx]  # (B, T, hidden_dim)

        # VQ-like cosine loss.
        commit_loss = 1.0 - (phrase_agg_n * hard_keys.detach()).sum(dim=-1)
        codebook_loss = 1.0 - (phrase_agg_n.detach() * hard_keys).sum(dim=-1)
        vq_like_loss = 0.5 * commit_loss + 0.5 * codebook_loss  # (B, T)

        if activity is not None:
            weight = activity.squeeze(-1).detach()
        else:
            weight = torch.ones(B, T, device=device, dtype=phrase_agg_n.dtype)

        if lengths is not None:
            valid = (
                torch.arange(T, device=device).unsqueeze(0)
                < lengths.unsqueeze(1)
            ).float()
            weight = weight * valid

        denom = weight.sum().clamp_min(1.0)
        vq_like_loss = (vq_like_loss * weight).sum() / denom

        # Per-position assignment entropy.
        # Since motif_probs is top-k sparse softmax, this encourages sharper top-k assignments.
        p = motif_probs.clamp_min(1e-8)
        per_pos_entropy = -(p * p.log()).sum(dim=-1)

        if lengths is not None:
            valid = (
                torch.arange(T, device=device).unsqueeze(0)
                < lengths.unsqueeze(1)
            ).float()
            per_pos_entropy = (per_pos_entropy * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            per_pos_entropy = per_pos_entropy.mean()

        # Usage entropy over the batch/time window.
        # Encourage not collapsing to a tiny number of motifs.
        if lengths is not None:
            valid = (
                torch.arange(T, device=device).unsqueeze(0)
                < lengths.unsqueeze(1)
            ).float()
            usage = (motif_probs * valid.unsqueeze(-1)).sum(dim=(0, 1))
            usage = usage / valid.sum().clamp_min(1.0)
        else:
            usage = motif_probs.mean(dim=(0, 1))

        usage = usage.clamp_min(1e-8)
        usage_entropy = -(usage * usage.log()).sum()

        # Coefficients deliberately small.
        loss = (
            0.1 * vq_like_loss
            + 0.001 * per_pos_entropy
            - 0.001 * usage_entropy
        )

        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: torch.Tensor,
        aux_steps: torch.Tensor,
        audio_emb: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ):
        """
        token_ids: (B, T)
        aux_steps: (B, T_aux, aux_dim)
        audio_emb: (B, T_audio, d_model)

        Returns:
            motif_seq:      (B, T, d_model)
            final_probs:    (B, T, n_motifs)
            final_activity: (B, T, 1)
            aux_loss:       scalar
        """
        B, T = token_ids.shape
        device = token_ids.device

        # 1. Local causal phrase encoding.
        phrase_agg = self.encode_multiscale_token_aux_windows(token_ids, aux_steps)
        phrase_agg_n = F.normalize(phrase_agg, dim=-1)

        # 2. Raw motif identity from token+aux.
        keys_n = F.normalize(self.motif_keys.weight, dim=-1)
        raw_logits = phrase_agg_n @ keys_n.T  # (B, T, n_motifs)

        # 3. Audio identity/activity prior.
        audio_aligned = self._align_audio(audio_emb, T)

        audio_h = self.audio_to_hidden(audio_aligned)
        audio_h = F.normalize(audio_h, dim=-1)
        audio_logits = audio_h @ keys_n.T

        combined_raw_logits = raw_logits + self.audio_identity_scale * audio_logits

        # 4. Raw sparse assignment.
        raw_probs = self._topk_softmax(combined_raw_logits)  # (B, T, n_motifs)
        raw_motif_vec = raw_probs @ self.motif_values.weight # (B, T, d_model)

        # 5. Raw activity.
        raw_activity_logit = (
            self.token_activity(phrase_agg)
            + self.audio_activity_scale * self.audio_activity(audio_aligned)
        )
        raw_activity = torch.sigmoid(raw_activity_logit)
        raw_probs_for_recent = raw_probs * raw_activity
        recent_vec = self.compute_recent_motif_vectors(raw_probs_for_recent)
        recent_vec = self.recent_proj(recent_vec)                      # (B, T, d_model)

        # 7. Motif evidence sequence.
        motif_evidence = (
            raw_motif_vec
            + self.phrase_to_model(phrase_agg)
            + recent_vec
        )

        # 8. Causal motif contextualization over time.
        causal_mask = make_causal_mask(T, device)
        pad_mask = make_key_padding_mask(lengths, T) if lengths is not None else None

        pos = torch.arange(T, device=device).unsqueeze(0)
        motif_ctx = motif_evidence + self.motif_pos(pos)
        for blk in self.motif_context_blocks:
            motif_ctx = blk(
                motif_ctx,
                attn_mask=causal_mask,
                key_padding_mask=pad_mask,
            )

        # 9. Context refines identity/activity.
        ctx_logits = self.context_to_logits(motif_ctx)
        ctx_activity_logit = self.context_to_activity(motif_ctx)

        final_logits = combined_raw_logits + ctx_logits
        final_activity = torch.sigmoid(raw_activity_logit + ctx_activity_logit)  # (B, T, 1)

        final_probs = self._topk_softmax(final_logits)                          # (B, T, n_motifs)

        # 10. Final motif sequence.
        motif_seq = final_probs @ self.motif_values.weight
        motif_seq = self.out_proj(motif_seq)

        # 11. Auxiliary codebook/usage loss.
        aux_loss = self._auxiliary_losses(
            phrase_agg_n=phrase_agg_n,
            motif_probs=final_probs,
            keys_n=keys_n,
            activity=final_activity,
            lengths=lengths,
        )

        return motif_seq, final_probs, final_activity, aux_loss