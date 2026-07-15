"""
Onset step-placement model for DDR/ITG chart generation.

Architecture:
  Encoder:
    Bidirectional Transformer over per-beat spectrogram CNN embeddings.
    Difficulty and BPM are injected as prepended [DIFF][BPM] context tokens.

  Heads:
    onset_head:
      Per-beat multi-label logits over 48 within-beat grid positions.

    density_head:
      Per-beat categorical distribution over number of onsets:
      {0, ..., density_max}.

Training:
  Phase 1:
    Train frozen diagnostic networks on ground-truth charts to predict:
      - difficulty conditioned on BPM
      - BPM bucket conditioned on difficulty

  Phase 2:
    Train OnsetModel with:
      - weighted BCE onset loss
      - categorical + expected-value density loss
      - frozen diagnostic regularization

Generation:
  Modes:
    threshold:
      sigmoid(onset_logits) > threshold

    density_topk:
      use density head to choose k_t per beat, then select top-k onset logits.

    density_viterbi:
      decode a smoothed density sequence with Viterbi, then select top-k onset logits.
"""

import csv, os, argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from util import *
from onset_generators import get_inputs_and_gens_onset

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# Utility: masks (causal + padding)
# -----------------------------

def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # disallow attending to future positions
    return mask  # shape (T, T), add to attention scores


def make_key_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    B = lengths.shape[0]
    if max_len is None:
        max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
    mask = arange >= lengths.unsqueeze(1)
    return mask  # (B, max_len) boolean

def get_grid_importance_weights(out_dim: int = 48, device: str = "cuda") -> torch.Tensor:
    # Initialize all positions with a baseline scale of 1.0
    weights = torch.ones(out_dim, device=device, dtype=torch.float32)
    
    # 16th Note / Downbeat Grid Anchors (High Priority)
    indices_16th = [0, 12, 24, 36]
    weights[indices_16th] = 2  #2.5 Scale factor for 16th grids
    
    # 24th Note / Triplet Grid Anchors (Moderate/High Priority)
    indices_24th = [8, 16, 32, 40]
    weights[indices_24th] = 1  #1.5 Scale factor for 24th grids
    
    # Optional: Suppress micro-timing ticks that human authors rarely use
    # (keeps the model from outputting random jitter)
    all_grid_positions = set(indices_16th + indices_24th + [6, 18, 30, 42]) # including 32nds
    for i in range(out_dim):
        if i not in all_grid_positions:
            weights[i] = 0.5  #0.5 De-emphasize unaligned micro-ticks
            
    return weights
# -----------------------------
# Positional Embedding
# -----------------------------
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return self.pos(positions)  # (B, T, d)
    


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OnsetConfig:
    n_channels: int = 3
    nframes: int = 32
    nfreq: int = 80
    cnn_hidden: int = 32

    d_model: int = 256
    n_heads: int = 8
    n_enc_layers: int = 8
    n_dec_layers: int = 4

    out_dim: int = 48
    max_beats: int = 2000
    n_difficulties: int = 100
    n_bpm_buckets: int = 37
    dropout: float = 0.1

    min_difficulty: float = 1.0
    max_difficulty: float = 50.0
    min_bpm: float = 40.0
    max_bpm: float = 400.0
# ──────────────────────────────────────────────────────────────────────────────
# Transformer utilities
# ──────────────────────────────────────────────────────────────────────────────

def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Returns a (T, T) upper-triangular additive mask with -inf on future positions."""
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


class LearnedPosEmb(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, d) → (B, T, d)
        B, T, _ = x.shape
        return self.emb(torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1))


class FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        h = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, d_model), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block with full (bidirectional) self-attention."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.RMSNorm(d_model)
        self.ffn  = FFN(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x


class NormlessEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block with full (bidirectional) self-attention."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn  = FFN(d_model, dropout)
        self.norm  = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        x = x + y
        x = x + self.ffn(self.norm(x))
        return x

class HierarchicalAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, frames_per_beat: int, beats_per_chunk: int, max_chunks: int = 2048):
        super().__init__()
        self.beats_per_chunk = beats_per_chunk
        self.frames_per_beat = frames_per_beat
        total_frames = beats_per_chunk * frames_per_beat
        
        # Intra-chunk frame position embedding (relative position within a single chunk)
        self.pos_emb = nn.Parameter(torch.randn(1, total_frames, d_model) * 0.02)
        
        # Macro-chunk position embedding (tells the block *which* chunk along the timeline it is)
        self.chunk_pos_emb = nn.Embedding(max_chunks, d_model)
        # Initialize with standard small scaling matching your parameter defaults
        nn.init.trunc_normal_(self.chunk_pos_emb.weight, std=0.02)
        
        self.attn_block = NormlessEncoderBlock(d_model, n_heads, dropout)
        self.aux_head = nn.Linear(frames_per_beat * d_model, 48)
        
        # Native, highly optimized PyTorch RMSNorm
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, h_beat: torch.Tensor, diff_tok: torch.Tensor, bpm_tok: torch.Tensor, B: int, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        D = h_beat.shape[-1]
        C = self.beats_per_chunk
        F_pb = self.frames_per_beat
        num_chunks = T // C
        
        # 1. Safe normalization — flat silence values map cleanly to uniform baselines
        h_beat = self.norm1(h_beat)
        
        # 2. Chunking
        # Reshape to group chunks along the batch dimension
        h_chunk = h_beat.reshape(B, num_chunks, C * F_pb, D).reshape(B * num_chunks, C * F_pb, D)
        
        # 3. Add Intra-Chunk Frame Position Embedding
        h_chunk = h_chunk + self.pos_emb
        
        # 4. Generate and Inject Macro Chunk Position Embedding
        # Create an array of indices [0, 1, 2, ..., num_chunks-1] matching the current song layout
        chunk_idx = torch.arange(num_chunks, device=h_beat.device, dtype=torch.long)
        # Look up positions -> shape: (num_chunks, d_model) -> expand to match the batch dimensions
        c_emb = self.chunk_pos_emb(chunk_idx) # (num_chunks, D)
        c_emb = c_emb.repeat(B, 1).unsqueeze(1) # (B * num_chunks, 1, D)
        
        # Broadcast add across the timeline frame dimension (C * F_pb)
        h_chunk = h_chunk + c_emb
        
        # 5. Context Interleaving
        #diff_ctx = diff_tok.repeat_interleave(num_chunks, dim=0)
        #bpm_ctx = bpm_tok.repeat_interleave(num_chunks, dim=0)
        
        # 6. Attention
        #h_chunk = torch.cat([diff_ctx, bpm_ctx, h_chunk], dim=1)
        #h_chunk = self.attn_block(h_chunk)[:, 2:, :]
        h_chunk = self.attn_block(h_chunk)
        
        # 7. Restore standard shape
        h_out = h_chunk.reshape(B, num_chunks, C, F_pb, D).reshape(B * T, F_pb, D)
        h_out = self.norm2(h_out)
        
        # 8. Auxiliary predictions
        logits = self.aux_head(h_out.reshape(B * T, -1)).view(B, T, 48)
        
        return h_out, logits

class GatedCompression(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Project to twice the output size to split into value and gate paths
        self.proj = nn.Linear(in_features, out_features * 2)
        self.norm = nn.RMSNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        projected = self.proj(x)
        
        # Split along the last dimension into values (v) and gates (g)
        v, g = projected.chunk(2, dim=-1)
        
        # The gate determines how much of each feature passes through (0.0 to 1.0)
        gated_output = v * torch.sigmoid(g)
        
        return self.norm(gated_output)


# ──────────────────────────────────────────────────────────────────────────────
# Per-beat CNN encoder
# ──────────────────────────────────────────────────────────────────────────────

class BeatCNNEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int, hidden: int = 32,
                 nframes: int = 32, nfreq: int = 80, use_norm: bool = False):
        super().__init__()

        self.hidden = hidden
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=(7, 3), stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), stride=(1,3) , padding=(1,0)),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=(3, 3), stride=1 , padding=0),
            nn.GELU(),
            nn.Conv2d(hidden*2, hidden*2, kernel_size=(3, 3), stride=(1,3) , padding=(1,0)),
            nn.GELU(),
        )
        self.slice_proj = nn.Sequential(
            nn.Linear(hidden*16, hidden*16),
            nn.GELU(),
            nn.Linear(hidden*16, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input format: (L*B, in_ch, nframes=32, nfreq=80)
        x = self.conv(x).permute(0, 2, 1, 3).reshape(-1, 24, self.hidden*16)
        x = self.slice_proj(x)
        return x


class ConvHead(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(0.1)
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = h.permute(0, 2, 1)
        h = self.conv(h)
        h = h.permute(0, 2, 1)
        
        return x + h

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic Network
# ──────────────────────────────────────────────────────────────────────────────
class HybridDiagnosticNetwork(nn.Module):
    def __init__(self, 
                 out_dim: int, 
                 d_model: int, 
                 n_classes: int, 
                 n_heads: int,
                 min_val: float,
                 max_val: float,
                 n_lstm_layers: int = 1,
                 n_attn_blocks: int = 1,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        # Store configuration bounds safely
        self.min_val = min_val
        self.max_val = max_val

        # 1. Local Pattern Extractor
        self.input_proj = nn.Linear(out_dim, d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Continuous Conditioning Projection Layer
        # Replacing categorical lookup with a smooth monotonic mapping MLP
        self.cond_continuous_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model // 2, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Macro Global Relations Engine (Self-Attention)
        self.pos_emb    = LearnedPosEmb(2001, d_model) 
        self.sa_block   = EncoderBlock(d_model, n_heads, dropout)
        self.ln         = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

        self.r_dropout = nn.Dropout(dropout)
        self.n_lstm_layers = n_lstm_layers
        self.n_attn_blocks = n_attn_blocks

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Coerce condition shape safely to (B, 1) and cast to float
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1)
        elif cond.dim() == 2 and cond.shape[1] != 1:
            cond = cond.view(-1, 1)
        cond = cond.to(dtype=torch.float32, device=x.device)

        # 1. Local texture extraction
        h = self.input_proj(x)
        h = h.permute(0, 2, 1)               
        h = self.conv(h)                     
        h = h.permute(0, 2, 1)               

        # 2. Strict Min-Max Normalization to bounded [0.0, 1.0] range
        cond_norm = (cond - self.min_val) / (self.max_val - self.min_val)
        cond_norm = cond_norm.clamp(0.0, 1.0)

        # 3. Process through Continuous MLP Map and element-wise inject
        c_emb = self.cond_continuous_proj(cond_norm).unsqueeze(1) # Shape: (B, 1, d_model)
        h = h + c_emb
        
        for _ in range(self.n_lstm_layers):                       
            h, _ = self.lstm(h)
            h = self.r_dropout(h)

        # 4. Macro global structural analysis via Self-Attention
        h = h + self.pos_emb(h)
        for _ in range(self.n_attn_blocks):
            h = self.sa_block(h)
            h = self.ln(h)
        
        # Global pooling and classification
        pooled_sequence = h.mean(dim=1)                # (B, d_model)
        return self.head(pooled_sequence)

# ──────────────────────────────────────────────────────────────────────────────
# Onset encoder-decoder model
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Refactored Hierarchical OnsetModel
# ──────────────────────────────────────────────────────────────────────────────
class OnsetModel(nn.Module):
    def __init__(self, cfg: OnsetConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Custom CNN encoder that leaves maps raw (no flat/proj)
        self.beat_cnn = BeatCNNEncoder(cfg.n_channels, d, cfg.cnn_hidden, cfg.nframes, cfg.nfreq)

        # Context injections for difficulty and BPM
        self.diff_continuous_proj = nn.Sequential(nn.Linear(1, d // 2), nn.GELU(), nn.Linear(d // 2, d))
        self.bpm_continuous_proj = nn.Sequential(nn.Linear(1, d // 2), nn.GELU(), nn.Linear(d // 2, d))

        # Reusable Hierarchical Blocks (Beats per chunk: 1, 4, 16, 32)
        self.level1 = HierarchicalAttnBlock(d, cfg.n_heads, cfg.dropout, frames_per_beat=24, beats_per_chunk=1)
        self.level2 = HierarchicalAttnBlock(d, cfg.n_heads, cfg.dropout, frames_per_beat=24, beats_per_chunk=4)
        self.level3 = HierarchicalAttnBlock(d, cfg.n_heads, cfg.dropout, frames_per_beat=24, beats_per_chunk=16)
        self.level4 = HierarchicalAttnBlock(d, cfg.n_heads, cfg.dropout, frames_per_beat=24, beats_per_chunk=32)

        # Level 5: 4x Frame Compression (24 frames -> 6 frames) within 64-beat chunks
        self.frame_compress = GatedCompression(4 * d, d)
        self.level5 = HierarchicalAttnBlock(d, cfg.n_heads, cfg.dropout, frames_per_beat=6, beats_per_chunk=64)

        # Level 6: Global Block compressing whole beats down to a single embedding vector
        self.beat_compress = GatedCompression(6 * d, d)
        self.level6_pos = nn.Parameter(torch.randn(1, cfg.max_beats+200, d) * 0.02)
        self.level6 = nn.ModuleList([EncoderBlock(d, cfg.n_heads, cfg.dropout)
                                         for _ in range(cfg.n_enc_layers)])
        self.conv_head = ConvHead(d)

        # Deepest primary target prediction heads
        self.onset_head = nn.Linear(d, cfg.out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _coerce(self, val, B: int, device, dtype=torch.float32):
        if not torch.is_tensor(val):
            val = torch.as_tensor(val, dtype=dtype, device=device)
        else:
            val = val.to(device=device, dtype=dtype)
        if val.dim() == 0:
            val = val.expand(B)
        elif val.shape == (B, 1):
            val = val.squeeze(-1)
        return val

    def forward(self, x: torch.Tensor, bpm, difficulty, target=None):
        B, T, Freq, W, C = x.shape
        device = x.device

        # 1. Format condition vectors into auxiliary context tokens
        diff_raw = self._coerce(difficulty, B, device)
        bpm_raw = self._coerce(bpm, B, device)
        diff_norm = ((diff_raw - self.cfg.min_difficulty) / (self.cfg.max_difficulty - self.cfg.min_difficulty)).clamp(0.0, 1.0).unsqueeze(-1)
        bpm_norm = ((bpm_raw - self.cfg.min_bpm) / (self.cfg.max_bpm - self.cfg.min_bpm)).clamp(0.0, 1.0).unsqueeze(-1)

        diff_tok = self.diff_continuous_proj(diff_norm).unsqueeze(1)  # (B, 1, d)
        bpm_tok = self.bpm_continuous_proj(bpm_norm).unsqueeze(1)    # (B, 1, d)

        # 2. Extract spectrogram slice blocks from the CNN
        xb = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, Freq, W)
        h = self.beat_cnn(xb)  # Out: (B * T, 64, 24, 8)

        # 3. Handle dynamic sequence safety padding to a multiple of 64
        pad_len = (64 - (T % 64)) % 64
        if pad_len > 0:
            # Pads along the sequence length dimension (T) after temporary reshaping
            h = h.view(B, T, 24, -1)
            h = F.pad(h, (0, 0, 0, 0, 0, pad_len))
            h = h.reshape(B * (T + pad_len), 24, -1)
        T_padded = T + pad_len

        # 4. Standardized Hierarchical Computations
        h, logits_lvl1 = self.level1(h, diff_tok, bpm_tok, B, T_padded)
        h, logits_lvl2 = self.level2(h, diff_tok, bpm_tok, B, T_padded)
        h, logits_lvl3 = self.level3(h, diff_tok, bpm_tok, B, T_padded)
        h, logits_lvl4 = self.level4(h, diff_tok, bpm_tok, B, T_padded)

        # 5. Level 5: 4x Frame Compression inside 64-beat chunks
        # Group adjacent 4 frames: (B * T_padded, 6, 4, d) -> (B * T_padded, 6, 4 * d) -> (B * T_padded, 6, d)
        h = h.view(B * T_padded, 6, 4, -1).reshape(B * T_padded, 6, -1)
        h = self.frame_compress(h)
        h, logits_lvl5 = self.level5(h, diff_tok, bpm_tok, B, T_padded)

        # 6. Level 6: Global Chart Architecture over uniform full beat embeddings
        # Condense 6 compressed frames into 1 master vector per beat: (B * T_padded, 6 * d) -> (B * T_padded, d)
        h_global = self.beat_compress(h.reshape(B * T_padded, -1)).view(B, T_padded, -1)
        h_global = h_global + self.level6_pos[:, :T_padded, :]
        
        h_global = torch.cat([diff_tok, bpm_tok, h_global], dim=1)
        for blk in self.level6:
            h_global = blk(h_global)

        h_global = h_global[:, 2:, :] # (B, T_padded, d)
        h_global = self.conv_head(h_global)

        # Final full resolution chart evaluation heads
        onset_logits = self.onset_head(h_global)

        # 7. Unpad tensors cleanly back to true structural length before returning
        return {
            "onset_logits": onset_logits[:, :T, :],
            "level1_logits": logits_lvl1[:, :T, :],
            "level2_logits": logits_lvl2[:, :T, :],
            "level3_logits": logits_lvl3[:, :T, :],
            "level4_logits": logits_lvl4[:, :T, :],
            "level5_logits": logits_lvl5[:, :T, :],
            "features": h_global[:, :T, :],
        }

    def predict_prob(self, x, bpm, difficulty):
        out = self.forward(x, bpm, difficulty)
        return torch.sigmoid(out["onset_logits"])
    
    def _bpm_to_bucket(self, bpm, B: int, device) -> torch.Tensor:
        bpm_f = self._coerce(bpm, B, device, dtype=torch.float32)
        bucket = ((bpm_f - 40.0) / 10.0).round().long()
        return bucket.clamp(0, self.cfg.n_bpm_buckets - 1)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        bpm,
        difficulty,
        threshold: float = 0.5,
        mode: str = "threshold",
        z_energy_thresh: float = -1.5, # Beats with mean log-energy below this are silence
    ) -> np.ndarray:
        """
        mode:
            "threshold"
        """
        B, T, Freq, W, C = x.shape
        assert B == 1, "Generation assumes batch size 1."

        out = self.forward(x, bpm, difficulty)
        onset_logits = out["onset_logits"].squeeze(0)          # (T, 48)

        if mode == "threshold":
            probs = torch.sigmoid(onset_logits)
            
            # --- Z-Score Silence Gate ---
            # x is Z-scored: 0.0 = song average, negative = quieter than average
            beat_energy = torch.mean(x.float(), dim=(2, 3, 4)).squeeze(0) # (1, T)
            print(beat_energy)
            
            # Any beat more than 1 std dev below the song's average is considered silence
            active_mask = (beat_energy > z_energy_thresh)
            
            probs = probs * active_mask.unsqueeze(-1).float()
            # ------------------------------

            return (probs > threshold).float().cpu().numpy()

        raise ValueError(f"Unknown generation mode: {mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Training and evaluation
# ──────────────────────────────────────────────────────────────────────────────

class OrdinalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, n_classes) - raw unnormalized network outputs
        targets: (B) - integer class/bucket labels (0 to n_classes-1)
        """
        # 1. Convert raw logits to a valid probability distribution
        probs = F.softmax(logits, dim=-1)
        
        # 2. Convert integer targets to one-hot vectors, e.g., 5 -> [0,0,0,0,0,1,0...]
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # 3. Compute Cumulative Distribution Functions (CDF) along the bucket dimension
        probs_cdf = torch.cumsum(probs, dim=-1)
        targets_cdf = torch.cumsum(targets_one_hot, dim=-1)
        
        # 4. Compute the structural distance penalty (MSE over the CDFs)
        # This penalizes the model based on HOW FAR away its probability mass is 
        # from the target bucket.
        loss = torch.pow(probs_cdf - targets_cdf, 2).sum(dim=-1).mean()
        return loss

def stream_loss(pred_logits, targets, window_size=4, active_mask=None):
    """
    Enforces stream continuity by penalizing the difference in rolling 
    step-density. Cannot go negative. 
    """
    probs = torch.sigmoid(pred_logits)
    
    # 1. Collapse 48-grid into a per-beat "density" scalar
    pred_density = probs.sum(dim=-1)  # (B, T)
    target_density = targets.sum(dim=-1)  # (B, T)
    
    # 2. Ignore silence to prevent penalizing missing steps in dead air
    if active_mask is not None:
        pred_density = pred_density * active_mask.float()
        target_density = target_density * active_mask.float()
    
    # 3. Smooth into rolling windows (e.g., 4 beats = 1 bar)
    padded_pred = F.pad(pred_density.unsqueeze(1), (window_size//2, window_size//2), mode='replicate')
    padded_tgt = F.pad(target_density.unsqueeze(1), (window_size//2, window_size//2), mode='replicate')
    
    smooth_pred = F.avg_pool1d(padded_pred, kernel_size=window_size, stride=1).squeeze(1)
    smooth_tgt = F.avg_pool1d(padded_tgt, kernel_size=window_size, stride=1).squeeze(1)
    
    # 4. Smooth L1 (Huber) loss. 
    # Penalizes density deficits proportionally, strictly >= 0.
    return F.smooth_l1_loss(smooth_pred, smooth_tgt, beta=1.0)

def evaluate_onset(model, generator, device, nsteps=100):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_probs = []
    losses = []

    with torch.no_grad():
        for _ in tqdm(range(nsteps), desc="Validating Direct Onset", leave=False):
            try:
                x, bpm, difficulty, target = next(generator)
            except StopIteration:
                break

            x = torch.tensor(np.array(x), dtype=torch.float, device=device).unsqueeze(0)
            target = torch.tensor(np.array(target), dtype=torch.float, device=device).unsqueeze(0)
            bpm = torch.tensor(bpm, dtype=torch.float, device=device)
            diff = torch.tensor([difficulty], dtype=torch.long, device=device)

            out = model(x, bpm, diff, target=target)
            logits = out["onset_logits"]

            losses.append(criterion(logits, target).item())

            all_probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
            all_targets.append(target.cpu().numpy().ravel())

    if not all_targets:
        return None

    y_true = np.concatenate(all_targets).astype(np.int32)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    return {
        "loss": float(np.mean(losses)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def train_onset(
    model,
    device="cuda",
    lr=1e-4,
    weight_decay=1e-2,
    accum_steps=1,
    patience=10,
    scheduler_patience=5,
    use_scheduler=False,
    use_early_stopping=False,
    stream_labels_fp="onset/songs/stream_labels.pkl",
    model_dir="trained_models",
    train_txt_fp="onset/songs/songs_train.txt",
    test_txt_fp="onset/songs/songs_test.txt",
    load_checkpoint=False,
    model_name="onset",
    nepochs=200,
    lambda_diag=0.005,
    checkpoint_metric="val_pr_auc",
):
    os.makedirs(model_dir, exist_ok=True)
    model.to(device)


    train_gen, test_gen, n_train, n_test = get_inputs_and_gens_onset(
        train_txt_fp, test_txt_fp, stream_labels_fp
    )

    # Instantiate Diagnostic networks.
    diag_diff = HybridDiagnosticNetwork(
        out_dim=model.cfg.out_dim,
        d_model=256,
        n_classes=model.cfg.n_difficulties,
        #n_cond_classes=model.cfg.n_bpm_buckets,
        min_val=1.0, max_val=50.0,
        n_heads=model.cfg.n_heads,
        dropout=model.cfg.dropout,
    ).to(device)

    diag_bpm = HybridDiagnosticNetwork(
        out_dim=model.cfg.out_dim,
        d_model=256,
        n_classes=model.cfg.n_bpm_buckets,
        #n_cond_classes=model.cfg.n_difficulties,
        min_val=30.0, max_val=400.0,
        n_heads=model.cfg.n_heads,
        dropout=model.cfg.dropout,
    ).to(device)

    diag_ckpt_fp = os.path.join(model_dir, f"{model_name}_diag_best.pt")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: Train Diagnostic Networks on Ground Truth Charts
    # ──────────────────────────────────────────────────────────────────────────
    if not load_checkpoint or not os.path.exists(diag_ckpt_fp):
        print("\n=== STARTING PHASE 1: Training Diagnostic Classifiers on Ground Truth ===")

        diag_opt = optim.AdamW(
            list(diag_diff.parameters()) + list(diag_bpm.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_diag_val_loss = float("inf")
        diag_patience_counter = 0
        criterion = OrdinalLoss()

        for d_epoch in range(1, nepochs + 1):
            diag_diff.train()
            diag_bpm.train()

            epoch_diag_losses = []
            pbar = tqdm(range(n_train), desc=f"Phase 1 - Epoch {d_epoch}", leave=False)

            diag_opt.zero_grad()

            for i in pbar:
                _, bpm, difficulty, target = next(train_gen)

                target = torch.tensor(
                    np.array(target),
                    dtype=torch.float,
                    device=device,
                ).unsqueeze(0)

                bpm_t = torch.tensor(bpm, dtype=torch.float, device=device)
                diff_t = torch.tensor([difficulty], dtype=torch.long, device=device)

                diff_v = model._coerce(diff_t, 1, device, dtype=torch.long)
                bpm_buck = model._bpm_to_bucket(bpm_t, 1, device)

                pred_diff = diag_diff(target, bpm_buck)
                pred_bpm = diag_bpm(target, diff_v)

                #loss_d = (
                #    F.cross_entropy(pred_diff, diff_v)
                #    + F.cross_entropy(pred_bpm, bpm_buck)
                #)

                loss_d = (
                    criterion(pred_diff, diff_v)
                    + criterion(pred_bpm, bpm_buck)
                )

                loss_d = loss_d / accum_steps
                loss_d.backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == n_train:
                    nn.utils.clip_grad_norm_(
                        list(diag_diff.parameters()) + list(diag_bpm.parameters()),
                        1.0,
                    )
                    diag_opt.step()
                    diag_opt.zero_grad()

                epoch_diag_losses.append(loss_d.item() * accum_steps)

            # Evaluate diagnostics.
            diag_diff.eval()
            diag_bpm.eval()

            val_diag_losses = []

            with torch.no_grad():
                for _ in range(n_test):
                    _, bpm, difficulty, target = next(test_gen)

                    target = torch.tensor(
                        np.array(target),
                        dtype=torch.float,
                        device=device,
                    ).unsqueeze(0)

                    bpm_t = torch.tensor(bpm, dtype=torch.float, device=device)
                    diff_t = torch.tensor([difficulty], dtype=torch.long, device=device)

                    diff_v = model._coerce(diff_t, 1, device, dtype=torch.long)
                    bpm_buck = model._bpm_to_bucket(bpm_t, 1, device)

                    pred_diff = diag_diff(target, bpm_buck)
                    pred_bpm = diag_bpm(target, diff_v)

                    #val_loss_d = (
                    #    F.cross_entropy(pred_diff, diff_v)
                    #    + F.cross_entropy(pred_bpm, bpm_buck)
                    #)
                    val_loss_d = (
                        criterion(pred_diff, diff_v)
                        + criterion(pred_bpm, bpm_buck)
                    )

                    val_diag_losses.append(val_loss_d.item())

            mean_val_diag_loss = float(np.mean(val_diag_losses))
            mean_train_diag_loss = float(np.mean(epoch_diag_losses))

            print(
                f"Phase 1 - Epoch {d_epoch:03d} | "
                f"Train Loss: {mean_train_diag_loss:.4f} | "
                f"Val Loss: {mean_val_diag_loss:.4f}"
            )

            if mean_val_diag_loss < best_diag_val_loss:
                best_diag_val_loss = mean_val_diag_loss
                diag_patience_counter = 0

                torch.save(
                    {
                        "diag_diff_state": diag_diff.state_dict(),
                        "diag_bpm_state": diag_bpm.state_dict(),
                    },
                    diag_ckpt_fp,
                )

                print(f"  --> Best diagnostic models saved (Val Loss: {best_diag_val_loss:.4f})")

            else:
                diag_patience_counter += 1

                if diag_patience_counter >= patience:
                    print("  --> Early stopping Phase 1 reached. Diagnostic training completed.")
                    break

    else:
        print(f"\n[Skipping Phase 1] Existing diagnostic weights discovered at: {diag_ckpt_fp}")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: Train Main Model using Frozen Diagnostics
    # ──────────────────────────────────────────────────────────────────────────
    print("\n=== STARTING PHASE 2: Training Generative OnsetModel ===")

    diag_ckpt = torch.load(diag_ckpt_fp, map_location=device, weights_only=False)
    diag_diff.load_state_dict(diag_ckpt["diag_diff_state"])
    diag_bpm.load_state_dict(diag_ckpt["diag_bpm_state"])

    # Keep diagnostics in train mode for cuDNN LSTM backward compatibility,
    # but freeze parameters and disable stochastic dropout.
    diag_diff.train()
    diag_bpm.train()

    for p in diag_diff.parameters():
        p.requires_grad = False
    for p in diag_bpm.parameters():
        p.requires_grad = False

    for m in list(diag_diff.modules()) + list(diag_bpm.modules()):
        if isinstance(m, nn.Dropout):
            m.eval()

    # CSV header.
    log_fp = os.path.join(model_dir, f"{model_name}_metrics.csv")

    if not (load_checkpoint and os.path.exists(log_fp)):
        with open(log_fp, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch",

                "train_loss",
                "train_bce",
                "train_diag",
                "train_pr_auc",

                "val_loss",
                "val_pr_auc",
                "val_precision",
                "val_recall",
                "val_f1",
            ])

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    grid_weights = get_grid_importance_weights(out_dim=model.cfg.out_dim, device=device)

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=0.5,
            patience=scheduler_patience,
        )
        if use_scheduler
        else None
    )

    best_ckpt = os.path.join(model_dir, f"{model_name}_hier_streamloss_best.pt")
    best_f1_ckpt = os.path.join(model_dir, f"{model_name}_hier_streamloss_best_f1.pt")
    start_epoch = 1
    best_score = -1.0
    best_f1 = -1.0

    if load_checkpoint and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])

        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("score", ckpt.get("val_pr_auc", -1.0))
        best_f1 = ckpt.get("val_f1", -1.0)

        print(
            f"Resumed main network from epoch {ckpt['epoch']} "
            f"({ckpt.get('checkpoint_metric', 'val_pr_auc')}={best_score:.4f})"
        )

    for epoch in range(start_epoch, nepochs + 1):
        model.train()

        epoch_losses = []
        epoch_bce_losses = []
        epoch_diag_losses = []

        all_probs = []
        all_targets = []

        opt.zero_grad()

        pbar = tqdm(
            range(n_train),
            desc=f"Phase 2 - Epoch {epoch}/{nepochs}",
            leave=False,
        )

        for i in pbar:
            x, bpm, difficulty, target = next(train_gen)

            x = torch.tensor(np.array(x), dtype=torch.float, device=device).unsqueeze(0)
            target = torch.tensor(np.array(target), dtype=torch.float, device=device).unsqueeze(0)

            bpm_t = torch.tensor(bpm, dtype=torch.float, device=device)
            diff_t = torch.tensor([difficulty], dtype=torch.long, device=device)

            out = model(x, bpm_t, diff_t, target=target)
            logits = out["onset_logits"]          # (B, T, 48)
            
            # Primary standard objectives
            raw_loss = criterion(logits, target)
            dice_loss = stream_loss(logits, target)
            weighted_loss = raw_loss * grid_weights.view(1, 1, -1)
            loss_main = weighted_loss.mean()


            # Frozen diagnostic loss regularization
            probs = torch.sigmoid(logits)
            diff_v = model._coerce(diff_t, x.shape[0], device, dtype=torch.long)
            bpm_buck = model._bpm_to_bucket(bpm_t, x.shape[0], device)
            hard = (probs > 0.5).float()
            diag_input = hard + probs - probs.detach() # Shape: (B, T, 48)
            
            # --- SILENCE GATE FOR DIAGNOSTIC LOSS ---
            # 1. FIX: Keep dim=0 intact by NOT using squeeze(0). 
            # Average over Freq (2), Width (3), and Channels (4)
            beat_energy = torch.mean(x.float(), dim=(2, 3, 4))  # Shape: (B, T)
            
            # 2. Create boolean mask using a fixed log threshold
            # Your silence padding is ~-10.0 to -36.0. Active music is usually > -7.0 or -8.0
            active_mask = (beat_energy > -1.5)  # Shape: (B, T)
            
            # 3. FIX: Unsqueeze the trailing channel dimension to make it (B, T, 1)
            # This allows it to broadcast across all 48 channels of (B, T, 48) smoothly
            active_mask = active_mask.unsqueeze(-1)  # Shape: (B, T, 1)
            
            # Zero out any steps the model tries to hide in silence.
            diag_input = diag_input * active_mask
            # -------------------------------------------

            pred_diff = diag_diff(diag_input, bpm_buck)
            pred_bpm = diag_bpm(diag_input, diff_v)
            loss_diag_diff = F.cross_entropy(pred_diff, diff_v)
            loss_diag_bpm = F.cross_entropy(pred_bpm, bpm_buck)
            loss_diag = loss_diag_diff + loss_diag_bpm

            # Aggregate total losses together
            loss = (
                loss_main 
                + lambda_diag * loss_diag
                + 0.05 * dice_loss
            )
            
            loss = loss / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == n_train:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            loss_val = loss.item() * accum_steps

            epoch_losses.append(loss_val)
            epoch_bce_losses.append(loss_main.item())
            epoch_diag_losses.append(loss_diag.item())

            if i % 100 == 0:
                postfix = {
                    "loss": f"{np.mean(epoch_losses[-100:]):.4f}",
                    "bce": f"{np.mean(epoch_bce_losses[-100:]):.4f}",
                    "diag": f"{np.mean(epoch_diag_losses[-100:]):.3f}",
                }

                pbar.set_postfix(postfix)

            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
            all_targets.append(target.cpu().numpy().ravel())

        train_pr_auc = average_precision_score(
            np.concatenate(all_targets).astype(np.int32),
            np.concatenate(all_probs),
        )

        val_metrics = evaluate_onset(model, test_gen, device, nsteps=n_test)

        print(
            f"Epoch {epoch:03d} -----------------------------------------------------\n"
            f"  TRAIN | Loss: {np.mean(epoch_losses):.4f} [BCE: {np.mean(epoch_bce_losses):.4f}, Diag: {np.mean(epoch_diag_losses):.4f}] | PR-AUC: {train_pr_auc:.4f}\n"
            f"  VAL   | Loss: {val_metrics['loss']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}\n"
            f"------------------------------------------------------------------------"
        )

        with open(log_fp, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,

                np.mean(epoch_losses),
                np.mean(epoch_bce_losses),
                np.mean(epoch_diag_losses),
                train_pr_auc,

                val_metrics["loss"],
                val_metrics["pr_auc"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
            ])

        if scheduler:
            scheduler.step(val_metrics["pr_auc"])

        current_score = val_metrics["pr_auc"]

        if current_score is not None and current_score > best_score:
            best_score = current_score

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "epoch": epoch,
                    "score": current_score,
                    "checkpoint_metric": checkpoint_metric,
                    "val_pr_auc": val_metrics["pr_auc"],
                    "cfg": model.cfg,
                    "val_f1": val_metrics["f1"],
                },
                best_ckpt,
            )

            print(f"  --> Best model saved ({checkpoint_metric}: {best_score:.4f})")
        if val_metrics["f1"] is not None and val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "epoch": epoch,
                    "score": current_score,
                    "checkpoint_metric": checkpoint_metric,
                    "val_pr_auc": val_metrics["pr_auc"],
                    "cfg": model.cfg,
                    "val_f1": val_metrics["f1"],
                },
                best_f1_ckpt,
            )

            print(f"  --> Best model saved (val_f1: {best_f1:.4f})")

    torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}_final.pt"))


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDR Onset Model with Frozen Diagnostic Probes")
    
    # Data Paths
    parser.add_argument("--train_txt", type=str, default="onset/songs/songs_train.txt")
    parser.add_argument("--test_txt",  type=str, default="onset/songs/songs_test.txt")
    parser.add_argument("--labels",    type=str, default="onset/songs/stream_labels.pkl")
    
    # Training Hyperparameters
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--accum_steps", type=int,   default=32, help="Gradient accumulation steps")
    parser.add_argument("--wd",          type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--patience",    type=int,   default=10,   help="Patience limit for tracking Early Stopping")
    parser.add_argument("--lambda_diag", type=float, default=0.000, help="Regularization multiplier for frozen diagnostic networks")
    
    # Model Config
    parser.add_argument("--d_model",     type=int, default=256)
    parser.add_argument("--enc_layers",  type=int, default=4)
    parser.add_argument("--dec_layers",  type=int, default=4)
    
    # Misc
    parser.add_argument("--model_name", type=str, default="ddr_onset_v1")
    parser.add_argument("--model_dir",  type=str, default="trained_models")
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",     action="store_true", help="Resume phase 2 from best checkpoint")

    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        default="val_pr_auc",
        choices=["val_pr_auc", "density_topk_f1", "density_viterbi_f1"],
    )
    parser.add_argument(
    "--no_rel_pos",
    action="store_true",
    help="Disable relative position bias in onset encoder attention.",
    )

    parser.add_argument(
        "--rel_pos_buckets",
        type=int,
        default=32,
        help="Number of magnitude buckets per direction for relative position bias.",
    )

    parser.add_argument(
        "--rel_pos_max_distance",
        type=int,
        default=256,
        help="Maximum relative distance for log-bucketed relative position bias.",
    )

    args = parser.parse_args()

    config = OnsetConfig(
        d_model=args.d_model,
        n_enc_layers=args.enc_layers,
        n_dec_layers=args.dec_layers,
    )

    model = OnsetModel(config)

    train_onset(
        model=model,
        device=args.device,
        lr=args.lr,
        weight_decay=args.wd,
        accum_steps=args.accum_steps,
        patience=args.patience,
        nepochs=args.epochs,
        model_name=args.model_name,
        model_dir=args.model_dir,
        use_scheduler=True,
        use_early_stopping=True,
        scheduler_patience=5,
        stream_labels_fp=args.labels,
        train_txt_fp=args.train_txt,
        test_txt_fp=args.test_txt,
        load_checkpoint=args.resume,
        lambda_diag=args.lambda_diag,
        checkpoint_metric=args.checkpoint_metric,
    )