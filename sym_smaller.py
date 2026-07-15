import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import numpy as np
import argparse
import math
import shutil
from typing import Optional, Tuple
from dataclasses import dataclass

from vector_quantize_pytorch import ResidualVQ

from sym_pretrainer import *
from sym_generators import get_inputs_and_gens_sym
from sym_config import ModelConfig
from sym_encoders import *

TOPK_LIST = [1, 2, 3, 4, 5]  

# -----------------------------
# Utility: masks (causal + padding)
# -----------------------------
def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # disallow attending to future positions
    return mask  # shape (T, T)


def make_key_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    B = lengths.shape[0]
    if max_len is None:
        max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
    mask = arange >= lengths.unsqueeze(1)
    return mask  # (B, max_len) boolean


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
        return self.pos(positions)


# -----------------------------
# Transformer building blocks
# -----------------------------

class LayerFiLM(nn.Module):
    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, target_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T, target_dim), cond: (B, T, cond_dim)
        gamma_beta = self.proj(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return (1.0 + gamma) * x + beta
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class SelfAttentionBlock(nn.Module):
    """
    Pre-norm Transformer block with causal self-attention (GPT-style) + relative position bias.
    Modified to interleave FiLM conditioning right before normalization steps.
    """
    def __init__(self, d_model: int, n_heads: int, max_len: int = 2048, dropout: float = 0.0, num_buckets: int = 32, cond_dim: Optional[int] = None):
        super().__init__()
        self.cond_dim = cond_dim
        
        # Initialize FiLM layers if conditioning dimension is provided
        if self.cond_dim is not None:
            self.film1 = LayerFiLM(cond_dim=cond_dim, target_dim=d_model)
            self.film2 = LayerFiLM(cond_dim=cond_dim, target_dim=d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)

        self.num_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_len
        self.rel_pos_bias = nn.Embedding(2 * num_buckets, n_heads)

    def _relative_position_bucket(self, relative_positions: torch.Tensor) -> torch.Tensor:
        sign = (relative_positions < 0).long()
        n = torch.abs(relative_positions)

        max_exact = self.num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            (torch.log(n.float() / max_exact) / math.log(self.max_distance / max_exact))
            * (self.num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, self.num_buckets - 1))

        buckets = torch.where(is_small, n, val_if_large)
        return buckets * 2 + sign  

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Apply FiLM Conditioning before LayerNorm 1 if cond sequence is available
        h = x
        if self.cond_dim is not None and cond is not None:
            h = self.film1(h, cond)
        h = self.ln1(h)

        pos = torch.arange(T, device=x.device)
        rel_pos = pos[None, :] - pos[:, None]
        buckets = self._relative_position_bucket(rel_pos)
        rel_bias = self.rel_pos_bias(buckets).permute(2, 0, 1)  

        rel_bias_expanded = rel_bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.num_heads, T, T)

        if attn_mask is None:
            combined_mask = rel_bias_expanded
        else:
            combined_mask = attn_mask.unsqueeze(0) + rel_bias_expanded

        y, _ = self.attn(
            h, h, h,
            attn_mask=combined_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + y
        
        # Apply FiLM Conditioning before LayerNorm 2 (FeedForward Pre-norm)
        h_ff = x
        if self.cond_dim is not None and cond is not None:
            h_ff = self.film2(h_ff, cond)
            
        x = x + self.ff(self.ln2(h_ff))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, kv_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        q = self.ln_q(x_q)
        kv = self.ln_kv(x_kv)
        y, _ = self.attn(q, kv, kv, key_padding_mask=kv_padding_mask, need_weights=False)
        x = x_q + y
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_self_pre: int = 6      
    n_self_audio_pre: int = 2 
    n_self_post: int = 6     
    n_cross: int = 2         
    dropout: float = 0.1
    max_len: int = 2048
    num_classes: int = 256   
    aux_dim: int = 2         
    VQ_dim: int = 128        
    VQ_codes: int = 512      
    conv_dim: int = 32
    pred_steps: int = 4  
    conv_layers: int = 4
    conv_heads: int = 4
    tok_dropout: float = 0.2
    latent_vq: bool = False
    tok_vq_codes: int = 512
    tok_vq_num_quantizers: int = 4
    tok_vq_weight: float = 0.25
    motif_mode: str = "none"
    n_motifs: int = 256
    motif_window_sizes: tuple = (4, 6, 8, 12)
    motif_phrase_dim: int = 32
    motif_hidden_dim: int = 128
    motif_n_heads: int = 4
    motif_conf_threshold: float = 0.3
    motif_lambda: float = 0.01
    motif_recon_lambda: float = 0.1


class AuxEncoder(nn.Module):
    def __init__(self, aux_dim: int, d_model: int, dropout: float = 0.1, bucket_boundaries=None):
        super().__init__()
        if bucket_boundaries is None:
            bucket_boundaries = [0.13, 0.17, 0.26, 0.34, 0.51, 0.76, 1.01, 2.01, 4.01, 8.01, 16.01, 32.01]

        self.aux_dim = aux_dim
        self.d_model = d_model
        
        # Register boundaries as a buffer so it moves to GPU automatically
        self.register_buffer("boundaries", torch.tensor(bucket_boundaries, dtype=torch.float32))

        n_buckets = len(bucket_boundaries) + 1
        self.last_bucket_emb = nn.Embedding(n_buckets, d_model)
        self.next_bucket_emb = nn.Embedding(n_buckets, d_model)

        # Mixes down the 2 concatenated embedding representations into d_model
        self.mix = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aux: Tensor of shape (..., aux_dim) where aux[..., 0] is last_gap 
                 and aux[..., 1] is next_gap (if aux_dim > 1).
        Returns:
            Mixed auxiliary embedding of shape (..., d_model)
        """
        aux = aux.float()
        
        # 1. Bucketize and embed last_gap
        last_gap = aux[..., 0].contiguous()
        last_idx = torch.bucketize(last_gap, self.boundaries)
        last_emb = self.last_bucket_emb(last_idx)

        # 2. Bucketize and embed next_gap if it exists
        if self.aux_dim > 1:
            next_gap = aux[..., 1].contiguous()
            next_idx = torch.bucketize(next_gap, self.boundaries)
            next_emb = self.next_bucket_emb(next_idx)
        else:
            # Fallback to zero-index bucket if only 1 aux dimension is provided
            next_emb = self.next_bucket_emb(torch.zeros_like(last_idx))

        # 3. Cleanly mix only the discrete embedding information
        mixed_features = torch.cat([last_emb, next_emb], dim=-1)
        return self.mix(mixed_features)



class FiLMBlock(nn.Module):
    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        # Predicts both gamma (scale) and beta (shift) in a single linear layer
        self.proj = nn.Linear(cond_dim, target_dim * 2)
        
        # Initialize gamma close to 1 and beta close to 0 so training starts stably
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Expected shapes: x: (B, T, target_dim), cond: (B, T, cond_dim)
        gamma_beta = self.proj(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        # We add 1 to gamma so that a zero-initialized network yields a scale factor of 1
        return (1.0 + gamma) * x + beta


class GPTStyleAudioModel(nn.Module):
    def __init__(self, in_audio_channels, config):
        super().__init__()
        self.cfg = config
        self.pred_steps = config.pred_steps
        
        self.tok_drop_p_max = config.tok_dropout
        self.tok_drop_p = 0.0

        #self.audio_enc = DDCStepConvEncoder(
        #    in_channels=in_audio_channels,
        #    d_model=config.d_model,
        #    vq_dim=config.VQ_dim,
        #    num_codes=config.VQ_codes,
        #    n_layers=getattr(config, "conv_layers", 1),
        #    n_heads=getattr(config, "conv_heads", 4),
        #    do_vq=True
        #)

        self.audio_enc = BasicEncoder(
           in_channels=in_audio_channels,
            d_model=config.d_model,
        )

        self.audio_pos = LearnedPositionalEmbedding(config.max_len, config.d_model)
        self.audio_drop = nn.Dropout(config.dropout)

        self.d_pre = config.d_model // 2
        self.pre_heads = config.n_heads // 2

        self.aux_token_enc = AuxEncoder(config.aux_dim, self.d_pre, config.dropout)
        self.aux_audio_enc = AuxEncoder(config.aux_dim, config.d_model, config.dropout)

        self.tok_ln = nn.LayerNorm(self.d_pre)
        self.tok_gate_drop = nn.Dropout(config.dropout)

        self.vocab_size = config.num_classes + 1
        self.bos_id = config.num_classes

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_pre)
        self.tok_pos = LearnedPositionalEmbedding(config.max_len, self.d_pre)

        # Pass cond_dim into the Self-Attention blocks to activate inside-layer FiLM modifications
        self.token_self_pre = nn.ModuleList([
            SelfAttentionBlock(self.d_pre, self.pre_heads, max_len=config.max_len, dropout=config.dropout, cond_dim=self.d_pre)
            for _ in range(config.n_self_pre)
        ])

        self.audio_self_pre = nn.ModuleList([
            SelfAttentionBlock(config.d_model, config.n_heads, max_len=config.max_len, dropout=config.dropout, cond_dim=config.d_model)
            for _ in range(getattr(config, "n_self_audio_pre", 2))
        ])

        self.proj_up = nn.Linear(self.d_pre, config.d_model)

        self.kv_ln = nn.LayerNorm(config.d_model)
        self.kv_drop = nn.Dropout(config.dropout)

        self.cross_interleaved_self = nn.ModuleList([
            SelfAttentionBlock(config.d_model, config.n_heads, max_len=config.max_len, dropout=config.dropout)
            for _ in range(config.n_cross)
        ])
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_cross)
        ])
        self.token_self_post = nn.ModuleList([
            SelfAttentionBlock(config.d_model, config.n_heads, max_len=config.max_len, dropout=config.dropout)
            for _ in range(config.n_self_post)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes * config.pred_steps)

    def encode_audio(self, audio_steps, aux_steps, pad_mask_audio=None):
        # 1. Base Convolutions & Vector Quantization
        audio_emb, vq_loss = self.audio_enc(audio_steps)
        audio_emb = audio_emb + self.audio_pos(audio_emb)
        audio_emb = self.audio_drop(audio_emb)

        # 2. Extract Auxiliary Sequence 
        aux_emb_audio = self.aux_audio_enc(aux_steps)

        # 3. Interpolate aux time to match audio CNN time map
        if aux_emb_audio.size(1) != audio_emb.size(1):
            aux_emb_audio = aux_emb_audio.permute(0, 2, 1)  
            aux_emb_audio = nn.functional.interpolate(
                aux_emb_audio, 
                size=audio_emb.size(1), 
                mode='linear', 
                align_corners=False
            )
            aux_emb_audio = aux_emb_audio.permute(0, 2, 1)  

        kv = self.kv_ln(audio_emb)
        kv = self.kv_drop(kv)

        # 4. Injected Pervasively inside the audio layers
        for blk in self.audio_self_pre:
            kv = blk(kv, attn_mask=None, key_padding_mask=pad_mask_audio, cond=aux_emb_audio)

        return kv, vq_loss
    
    def forward(self, audio_steps, aux_steps, token_ids, lengths=None, preencoded_audio=None):
        B, T_tok = token_ids.size(0), token_ids.size(1)
        device = token_ids.device

        if preencoded_audio is not None:
            kv, enc_loss = preencoded_audio
            T_audio = kv.size(1)
        else:
            T_audio = audio_steps.size(1)
            
        pad_mask_audio = make_key_padding_mask(torch.full((B,), T_audio, device=device, dtype=torch.long), T_audio)

        # 1. Resolve Audio Pre-encoding Context
        if preencoded_audio is not None:
            kv, enc_loss = preencoded_audio
        else:
            kv, enc_loss = self.encode_audio(audio_steps, aux_steps, pad_mask_audio=pad_mask_audio)

        # Slice aux raw details to match target step frames
        if aux_steps.size(1) >= T_tok:
            aux_tok = aux_steps[:, :T_tok, :]
        else:
            pad_len = T_tok - aux_steps.size(1)
            aux_tok = torch.cat([aux_steps, aux_steps[:, -1:, :].expand(-1, pad_len, -1)], dim=1)

        # --- ALIGNED TOKEN AND AUX DROPOUT ---
        if self.training and self.tok_drop_p > 0:
            # Generate the layout drop mask
            drop_mask = torch.rand_like(token_ids.float()) < self.tok_drop_p
            drop_mask[:, 0] = False # Safe-keep BOS
            
            # Mask out token indices
            token_ids = token_ids.masked_fill(drop_mask, self.bos_id)
            
            # Simultaneously mask out aux_tok values at identical temporal positions
            # Setting dropped structural contexts to 0.0 forces reliance back on the audio stream
            aux_tok = aux_tok.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        # Extract embeddings
        aux_emb_token = self.aux_token_enc(aux_tok)
        tok_x = self.tok_emb(token_ids)
        tok_x = tok_x + self.tok_pos(tok_x)

        tok_x = self.tok_ln(tok_x)
        tok_x = self.tok_gate_drop(tok_x)

        # Token processing masks
        attn_mask = make_causal_mask(T_tok, device)
        pad_mask_tok = make_key_padding_mask(lengths, T_tok) if lengths is not None else None

        # Pervasive conditional mapping passed on directly per layer here
        for blk in self.token_self_pre:
            tok_x = blk(tok_x, attn_mask=attn_mask, key_padding_mask=pad_mask_tok, cond=aux_emb_token)

        tok_x = self.proj_up(tok_x)

        # Cross attention streams
        for cross_blk, self_blk in zip(self.cross_blocks, self.cross_interleaved_self):
            tok_x = cross_blk(tok_x, kv, kv_padding_mask=pad_mask_audio)
            tok_x = self_blk(tok_x, attn_mask=attn_mask, key_padding_mask=pad_mask_tok)

        # Post-cross decision layers
        for blk in self.token_self_post:
            tok_x = blk(tok_x, attn_mask=attn_mask, key_padding_mask=pad_mask_tok)

        # Decode head projection
        tok_x = self.ln_f(tok_x)
        logits = self.head(tok_x).view(B, T_tok, self.pred_steps, self.cfg.num_classes)

        return logits, enc_loss
    
    def generate(
        self,
        audio_steps=None,
        aux_steps=None,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        preencoded_audio=None,
        lengths: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        progress_callback=None,
        prefix_tokens: Optional[np.ndarray] = None,
        # --- Advanced Phrase Repetition Adjustments ---
        phrase_penalty: float = 1.07,      
        max_ngram: int = 8,                
        min_ngram: int = 4,                
        recency_window: int = 20,          
    ):
        """
        Autoregressive generation stepping completely through the chunk window.
        """
        def _nucleus_sample(logits, top_p=0.9, temperature=1.0):
            probs = F.softmax(logits / (temperature if temperature > 0 else 1e-8), dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            idx = torch.multinomial(sorted_probs, 1)
            return sorted_idx.gather(-1, idx).squeeze(-1)

        # === Device resolution ===
        if device is None:
            if preencoded_audio is not None:
                device = preencoded_audio[0].device
            elif isinstance(audio_steps, torch.Tensor):
                device = audio_steps.device
            else:
                device = next(self.parameters()).device

        # === Ensure clean numpy → tensor conversion ===
        if audio_steps is not None:
            if isinstance(audio_steps, list):
                audio_steps = np.array(audio_steps)
            if isinstance(audio_steps, np.ndarray):
                audio_steps = torch.from_numpy(audio_steps).float()
            audio_steps = audio_steps.to(device)

        if aux_steps is not None:
            if isinstance(aux_steps, list):
                aux_steps = np.array(aux_steps)
            if isinstance(aux_steps, np.ndarray):
                aux_steps = torch.from_numpy(aux_steps).float()
            aux_steps = aux_steps.to(device)

        # Encode audio once
        if preencoded_audio is not None:
            audio_emb, _ = preencoded_audio
        else:
            audio_emb, _ = self.encode_audio(audio_steps, aux_steps)

        B, T_audio, _ = audio_emb.shape
        T = min(max_len or T_audio, T_audio)

        if lengths is not None:
            lengths = torch.clamp(lengths.to(device), max=T)

        # === Initialize tokens with BOS and optional context prefix ===
        if prefix_tokens is not None and len(prefix_tokens) > 0:
            pt_tensor = torch.as_tensor(prefix_tokens, dtype=torch.long, device=device)
            if pt_tensor.ndim == 1:
                pt_tensor = pt_tensor.unsqueeze(0)
            if pt_tensor.shape[0] != B:
                pt_tensor = pt_tensor.expand(B, -1)
            
            num_prefix_tokens = pt_tensor.shape[1]
            bos_tensor = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
            tokens = torch.cat([bos_tensor, pt_tensor], dim=1)
        else:
            tokens = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
            num_prefix_tokens = 0

        # Track exactly how many tokens we append in this call
        generated_count = 0

        with torch.no_grad():
            for i in range(T):
                # Pass tokens forward entirely without shifting/slicing history window
                logits, _ = self.forward(
                    None,
                    aux_steps[:, :tokens.size(1)] if aux_steps is not None else None,
                    tokens,
                    lengths=torch.tensor([tokens.size(1)], device=device),
                    preencoded_audio=(audio_emb[:, :tokens.size(1)], 0.0),
                )
                step_logits = logits[:, -1, 0, :].clone()

                # --- Phrase Repetition Penalty Logic ---
                if phrase_penalty != 1.0 and (i + num_prefix_tokens) > min_ngram:
                    for b in range(B):
                        history = tokens[b, 1:].tolist()
                        history_len = len(history)
                        
                        window_start = max(0, history_len - recency_window)
                        context = history[window_start:]
                        context_len = len(context)

                        for n in range(min_ngram, min(max_ngram + 1, context_len + 1)):
                            current_prefix = context[-(n - 1):]
                            
                            for idx in range(context_len - n):
                                match_candidate = context[idx : idx + (n - 1)]
                                if match_candidate == current_prefix:
                                    forbidden_token = context[idx + (n - 1)]
                                    
                                    scale = phrase_penalty
                                    if step_logits[b, forbidden_token] > 0:
                                        step_logits[b, forbidden_token] /= scale
                                    else:
                                        step_logits[b, forbidden_token] *= scale

                # Sample or argmax
                if temperature <= 0 or top_p is None:
                    next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
                else:
                    next_token = _nucleus_sample(step_logits, top_p=top_p, temperature=temperature).unsqueeze(-1)

                tokens = torch.cat([tokens, next_token], dim=1)
                generated_count += 1

                if progress_callback is not None:
                    progress_callback(1)

        # Slice backwards from the end to get exactly the number of steps we generated
        tokens_np = tokens[:, -generated_count:].squeeze(0).detach().cpu().numpy()
        return tokens_np


# -----------------------------
# Loss helper
# -----------------------------
def sequence_multi_step_loss(
    logits: torch.Tensor,        
    targets: torch.Tensor,       
    lengths: Optional[torch.Tensor] = None,
    step_weights: Optional[torch.Tensor] = None,  
) -> torch.Tensor:
    B, T, pred_steps, C = logits.shape
    device = logits.device

    if step_weights is None:
        step_weights = torch.ones(pred_steps, device=device) / pred_steps
    else:
        step_weights = step_weights.to(device)
        step_weights = step_weights / step_weights.sum()

    total_loss = 0.0
    total_weight = 0.0

    for k in range(pred_steps):
        if T - k <= 0:
            continue

        step_logits = logits[:, :-k or None, k, :]   
        step_targets = targets[:, k:]                

        step_loss = F.cross_entropy(
            step_logits.reshape(-1, C),
            step_targets.reshape(-1),
            reduction="none",
            label_smoothing=0.05
        ).reshape(B, T - k)

        if lengths is not None:
            mask = ~make_key_padding_mask(lengths - k, T - k)  
            denom = mask.float().sum().clamp_min(1.0)
            step_loss = (step_loss * mask.float()).sum() / denom
        else:
            step_loss = step_loss.mean()

        w = step_weights[k]
        total_loss += w * step_loss
        total_weight += w

    return total_loss / max(total_weight, 1e-8) if total_weight > 0 else torch.tensor(0.0, device=device, requires_grad=True)


def step_weight_scheduler(global_step, total_steps, pred_steps, base_power=1.0):
    progress = min(global_step / total_steps, 1.0)
    weights = torch.zeros(pred_steps)
    for s in range(pred_steps):
        step_power = base_power * (1.0 + s / pred_steps)
        step_progress = progress ** step_power
        weights[s] = (1 - step_progress) * (1.0 if s == 0 else 0.0) + step_progress * 1.0

    return weights / weights.sum()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


 

def train_one_epoch(model, optimizer, train_gen, device, steps_per_epoch,
                    log_interval=100, lambda_vq=.25, epoch=1, total_epochs=50,
                    save_dir=None, model_name="model", checkpoint_interval=5000,
                    scheduler=None, grad_accum_steps=32,
                    tok_dropout_warmup_steps=5000):
    model.train()
    
    total_loss = 0.0
    final_total_loss = 0.0
    pred_steps = model.pred_steps
    topk_correct = [[0] * len(TOPK_LIST) for _ in range(pred_steps)]
    total_counts  = [0 for _ in range(pred_steps)]

    total_train_steps = steps_per_epoch * total_epochs
    pbar = tqdm(range(steps_per_epoch), desc=f"Training (epoch {epoch})", leave=False)
    global_sample_offset = (epoch - 1) * steps_per_epoch
    optimizer_step_in_epoch = 0

    optimizer.zero_grad(set_to_none=True)

    for sample_step in pbar:
        ns2, ns0, ns1 = next(train_gen)
        ns2, ns0, ns1 = np.array(ns2), np.array(ns0), np.array(ns1)

        audio = torch.as_tensor(ns2, dtype=torch.float32).unsqueeze(0).to(device)
        aux   = torch.as_tensor(ns0, dtype=torch.float32).unsqueeze(0).to(device)

        target_ids_np = ns1.argmax(axis=-1).astype(np.int64) if ns1.ndim == 2 else ns1.astype(np.int64)
        T = int(target_ids_np.shape[0])
        
        token_in = np.full((1, T), fill_value=model.bos_id, dtype=np.int64)
        if T > 1:
            token_in[0, 1:] = target_ids_np[:-1]

        token_in_t = torch.as_tensor(token_in, dtype=torch.long).to(device)
        target_t   = torch.as_tensor(target_ids_np, dtype=torch.long).unsqueeze(0).to(device)
        lengths    = torch.tensor([audio.shape[1]], device=device)

        # Update dynamic token dropout tracking rate via model properties
        global_sample = global_sample_offset + sample_step
        if tok_dropout_warmup_steps > 0:
            model.tok_drop_p = model.tok_drop_p_max * min(1.0, global_sample / tok_dropout_warmup_steps)

        logits, enc_loss = model(audio, aux, token_in_t, lengths)

        step_weights  = step_weight_scheduler(global_sample, total_train_steps, model.pred_steps)

        raw_loss = sequence_multi_step_loss(logits, target_t, lengths=lengths, step_weights=step_weights) + lambda_vq * enc_loss
        (raw_loss / grad_accum_steps).backward()

        raw_loss_val = raw_loss.item()
        total_loss       += raw_loss_val
        final_total_loss += raw_loss_val

        with torch.no_grad():
            max_k = min(max(TOPK_LIST), logits.size(-1))
            for step_k in range(pred_steps):
                if T - step_k <= 0:
                    continue
                step_logits  = logits[:, :-step_k or None, step_k, :]  
                step_targets = target_t[:, step_k:]                     
                mask = ~(make_key_padding_mask(lengths - step_k, step_logits.size(1))[0])

                topk_idx = step_logits[0].topk(max_k, dim=-1).indices  
                tgt      = step_targets[0, mask]                        
                topk_m   = topk_idx[mask]                               

                total_counts[step_k] += mask.sum().item()
                for j, ks in enumerate(TOPK_LIST):
                    hits = (topk_m[:, :min(ks, max_k)] == tgt.unsqueeze(-1)).any(dim=-1)
                    topk_correct[step_k][j] += hits.sum().item()

        is_accum_boundary = (sample_step + 1) % grad_accum_steps == 0
        is_epoch_end      = (sample_step + 1) == steps_per_epoch

        if is_accum_boundary or is_epoch_end:
            remainder = (sample_step + 1) % grad_accum_steps
            if is_epoch_end and remainder != 0:
                rescale = grad_accum_steps / remainder
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(rescale)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            optimizer_step_in_epoch += 1
            global_optimizer_step = ((epoch - 1) * (steps_per_epoch // grad_accum_steps) + optimizer_step_in_epoch)

            if save_dir and (global_optimizer_step % checkpoint_interval == 0 or is_epoch_end):
                latest_path = os.path.join(save_dir, f"{model_name}_latest.pt")
                for f in os.listdir(save_dir):
                    if f.startswith(f"{model_name}_checkpoint_step") and f.endswith(".pt"):
                        os.remove(os.path.join(save_dir, f))

                ckpt_path = os.path.join(save_dir, f"{model_name}_checkpoint_step{global_optimizer_step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_optimizer_step": global_optimizer_step,
                    "global_sample_step": global_sample,
                }, ckpt_path)
                shutil.copy(ckpt_path, latest_path)
                os.remove(ckpt_path)

        if (sample_step + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{total_loss / log_interval:.4f}", opt_steps=optimizer_step_in_epoch)
            total_loss = 0.0

    topk_accs = [[topk_correct[step_k][j] / max(total_counts[step_k], 1) for j in range(len(TOPK_LIST))] for step_k in range(pred_steps)]
    return final_total_loss / steps_per_epoch, topk_accs


@torch.no_grad()
def evaluate(model, test_gen, device, eval_steps=100):
    model.eval()
    total_loss = 0.0
    pred_steps = model.pred_steps
    topk_correct = [[0] * len(TOPK_LIST) for _ in range(pred_steps)]
    total        = [0 for _ in range(pred_steps)]

    pbar = tqdm(range(eval_steps), desc="Evaluating", leave=False)
    for _ in pbar:
        ns2, ns0, ns1 = next(test_gen)
        ns2, ns0, ns1 = np.array(ns2), np.array(ns0), np.array(ns1)

        audio = torch.as_tensor(ns2, dtype=torch.float32).unsqueeze(0).to(device)
        aux   = torch.as_tensor(ns0, dtype=torch.float32).unsqueeze(0).to(device)
        target_ids_np = ns1.argmax(axis=-1).astype(np.int64) if ns1.ndim == 2 else ns1.astype(np.int64)

        T = int(target_ids_np.shape[0])
        token_in = np.full((1, T), fill_value=model.bos_id, dtype=np.int64)
        if T > 1:
            token_in[0, 1:] = target_ids_np[:-1]

        token_in_t = torch.as_tensor(token_in, dtype=torch.long).to(device)
        target_t = torch.as_tensor(target_ids_np, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([audio.shape[1]], device=device)

        logits, _ = model(audio, aux, token_in_t, lengths)
        total_loss += sequence_multi_step_loss(logits, target_t, lengths=lengths, step_weights=torch.ones(model.pred_steps, device=device)).item()

        max_k = min(max(TOPK_LIST), logits.size(-1))
        for step_k in range(pred_steps):
            if T - step_k <= 0:
                continue
            step_logits  = logits[:, :-step_k or None, step_k, :]
            step_targets = target_t[:, step_k:]
            mask = ~(make_key_padding_mask(lengths - step_k, step_logits.size(1))[0])

            topk_idx = step_logits[0].topk(max_k, dim=-1).indices  
            tgt      = step_targets[0, mask]
            topk_m   = topk_idx[mask]

            total[step_k] += mask.sum().item()
            for j, ks in enumerate(TOPK_LIST):
                hits = (topk_m[:, :min(ks, max_k)] == tgt.unsqueeze(-1)).any(dim=-1)
                topk_correct[step_k][j] += hits.sum().item()

    topk_accs = [[topk_correct[step_k][j] / max(total[step_k], 1) for j in range(len(TOPK_LIST))] for step_k in range(pred_steps)]
    return total_loss / eval_steps, topk_accs


def main():
    parser = argparse.ArgumentParser(description="Train GPT-style audio model")
    parser.add_argument("--train_fp", type=str, default='sym/songs/songs_train.txt')
    parser.add_argument("--test_fp", type=str, default='sym/songs/songs_test.txt')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="trained_models")
    parser.add_argument("--model_name", type=str, default="sym_small")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_self_pre", type=int, default=4)
    parser.add_argument("--n_self_post", type=int, default=4)
    parser.add_argument("--conv_layers", type=int, default=1)
    parser.add_argument("--conv_heads", type=int, default=4)
    parser.add_argument("--n_cross", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=256)
    parser.add_argument("--aux_dim", type=int, default=2)
    parser.add_argument("--audio_radius", type=int, default=10)
    parser.add_argument("--max_chunk", type=int, default=2000)
    parser.add_argument("--min_chunk", type=int, default=50)
    parser.add_argument("--VQ_dim", type=int, default=256)
    parser.add_argument("--VQ_codes", type=int, default=512)
    parser.add_argument("--pred_steps", type=int, default=4)
    parser.add_argument("--lambda_vq", type=float, default=0.25)
    parser.add_argument("--conv_dim", type=int, default=32)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--n_self_audio_pre", type=int, default=2)
    parser.add_argument("--latent_vq", action="store_true")
    parser.add_argument("--tok_vq_codes", type=int, default=512)
    parser.add_argument("--tok_vq_num_quantizers", type=int, default=4)
    parser.add_argument("--tok_vq_weight", type=float, default=0.25)
    parser.add_argument("--motif_mode", type=str, default="none", choices=["cross", "gate", "none"])
    parser.add_argument("--n_motifs", type=int, default=256)
    parser.add_argument("--motif_window_sizes", type=int, nargs="+", default=[4, 6, 8, 12])
    parser.add_argument("--motif_phrase_dim", type=int, default=32)
    parser.add_argument("--motif_hidden_dim", type=int, default=128)
    parser.add_argument("--motif_n_heads", type=int, default=4)
    parser.add_argument("--motif_conf_threshold", type=float, default=0.3)
    parser.add_argument("--motif_lambda", type=float, default=0.01)
    parser.add_argument("--motif_recon_lambda", type=float, default=0.1)
    parser.add_argument("--mode", type=str, default="finetune", choices=["pretrain", "finetune", "both"])
    parser.add_argument("--pre_token_steps", type=int, default=10000)
    parser.add_argument("--pre_audio_steps", type=int, default=5000)
    parser.add_argument("--pre_mask_start", type=float, default=0.02)
    parser.add_argument("--pre_mask_end", type=float, default=0.5)
    parser.add_argument("--pre_audio_mask_end", type=float, default=0.6)
    parser.add_argument("--tok_dropout", type=float, default=0.2)
    parser.add_argument("--tok_dropout_warmup_steps", type=int, default=100000,
                        help="Steps to linearly ramp token dropout from 0 to tok_dropout max (0 = no warmup)")
    args = parser.parse_args()

    train_gen, test_gen, train_len, test_len = get_inputs_and_gens_sym(
        args.train_fp, args.test_fp, audio_radius=args.audio_radius,
        max_chunk=args.max_chunk, min_chunk=args.min_chunk, use_diff=(args.aux_dim == 3)
    )

    cfg = ModelConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_self_pre=args.n_self_pre,
        n_self_post=args.n_self_post, n_self_audio_pre=args.n_self_audio_pre,
        dropout=args.dropout, max_len=2048, num_classes=args.num_classes,
        aux_dim=args.aux_dim, n_cross=args.n_cross, VQ_dim=args.VQ_dim,
        VQ_codes=args.VQ_codes, conv_dim=args.conv_dim, pred_steps=args.pred_steps,
        conv_layers=args.conv_layers, conv_heads=args.conv_heads, tok_dropout=args.tok_dropout,
        latent_vq=args.latent_vq, tok_vq_codes=args.tok_vq_codes,
        tok_vq_num_quantizers=args.tok_vq_num_quantizers, tok_vq_weight=args.tok_vq_weight,
        motif_mode=args.motif_mode, n_motifs=args.n_motifs, motif_window_sizes=tuple(args.motif_window_sizes),
        motif_phrase_dim=args.motif_phrase_dim, motif_hidden_dim=args.motif_hidden_dim,
        motif_n_heads=args.motif_n_heads, motif_conf_threshold=args.motif_conf_threshold,
        motif_lambda=args.motif_lambda, motif_recon_lambda=args.motif_recon_lambda,
    )
    
    model = GPTStyleAudioModel(in_audio_channels=3, config=cfg).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    total_optimizer_steps = (train_len // args.grad_accum_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_optimizer_steps), num_training_steps=total_optimizer_steps)

    os.makedirs(args.save_dir, exist_ok=True)

    best_ckpt_step0 = os.path.join(args.save_dir, f"{args.model_name}_best_step0.pt")
    best_ckpt_avg   = os.path.join(args.save_dir, f"{args.model_name}_best_avg.pt")
    final_path      = os.path.join(args.save_dir, f"{args.model_name}_final.pt")
    latest_ckpt     = os.path.join(args.save_dir, f"{args.model_name}_latest.pt")

    resume_path = None
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        resume_path = args.resume_ckpt
    elif os.path.isfile(latest_ckpt):
        resume_path = latest_ckpt
    elif os.path.isfile(best_ckpt_avg):
        resume_path = best_ckpt_avg

    start_epoch = 1
    best_val_acc_avg = 0.0
    best_val_acc_step0 = 0.0
    patience = args.patience
    epochs_no_improve = 0
    early_stop = False

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=args.device, weights_only=False)
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt.get("model", {}).items() if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_state, strict=True)

        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"Warning: restarting optimizer state due to config shift: {e}")

        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc_avg = ckpt.get("best_val_acc_avg", 0.0)
        best_val_acc_step0 = ckpt.get("best_val_acc_step0", 0.0)

    if args.mode in ("pretrain", "both"):
        print("\n=== Pretraining phase 1: token-only ===")
        pretrain_token_phase(model=model, optimizer=optimizer, train_gen=train_gen, device=args.device, steps=args.pre_token_steps, mask_start=args.pre_mask_start, mask_end=args.pre_mask_end, lambda_vq=0.0, log_interval=100, save_dir=args.save_dir, model_name=args.model_name, global_step_offset=0, checkpoint_interval=5000)
        print("\n=== Pretraining phase 2: audio introduced ===")
        pretrain_audio_phase(model=model, optimizer=optimizer, train_gen=train_gen, device=args.device, steps=args.pre_audio_steps, mask_start=args.pre_mask_start, mask_end=args.pre_audio_mask_end, lambda_vq=args.lambda_vq, log_interval=100, save_dir=args.save_dir, model_name=args.model_name, global_step_offset=args.pre_token_steps, checkpoint_interval=5000)
    
    if args.mode in ("finetune", "both"):
        for epoch in range(start_epoch, args.epochs + 1):
            if early_stop:
                print("Early stopping triggered.")
                break
            print(f"\n=== Epoch {epoch}/{args.epochs} ===")

            train_loss, train_accs = train_one_epoch(
                model, optimizer, train_gen, args.device, train_len, lambda_vq=args.lambda_vq,
                epoch=epoch, total_epochs=args.epochs, save_dir=args.save_dir, model_name=args.model_name,
                checkpoint_interval=5000, scheduler=scheduler, grad_accum_steps=args.grad_accum_steps,
                tok_dropout_warmup_steps=args.tok_dropout_warmup_steps
            )

            val_loss, val_accs = evaluate(model, test_gen, args.device, test_len)

            # top-1 accuracy for each pred step (used for checkpoint decisions)
            train_top1 = [train_accs[k][0] for k in range(len(train_accs))]
            val_top1   = [val_accs[k][0]   for k in range(len(val_accs))]
            train_acc      = train_top1[0]
            val_acc        = val_top1[0]
            avg_train_acc  = sum(train_top1) / len(train_top1)
            avg_val_acc    = sum(val_top1)   / len(val_top1)

            # ── Pretty-print per-step top-k breakdown ──────────────────────
            k_header = "  ".join(f"top{ks}" for ks in TOPK_LIST)
            print(f"\n{'':>12}  {k_header}")
            for step_k, row in enumerate(train_accs):
                row_str = "  ".join(f"{acc*100:>5.1f}%" for acc in row)
                print(f"  train @{step_k+1}:  {row_str}")
            for step_k, row in enumerate(val_accs):
                row_str = "  ".join(f"{acc*100:>5.1f}%" for acc in row)
                print(f"  val   @{step_k+1}:  {row_str}")

            print(
                f"Training   loss: {train_loss:.4f} | top-1 step-1: {train_acc*100:.2f}%  avg: {avg_train_acc*100:.2f}%\n"
                f"Validation loss: {val_loss:.4f} | top-1 step-1: {val_acc*100:.2f}%  avg: {avg_val_acc*100:.2f}%\n"
                f"tok_drop_p: {model.tok_drop_p:.4f} / {model.tok_drop_p_max:.4f}"
            )

            # Save best step-0 model
            if val_acc > best_val_acc_step0:
                best_val_acc_step0 = val_acc
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg,
                    "best_val_acc_step0": best_val_acc_step0,
                    "best_val_acc_avg": best_val_acc_avg
                }, best_ckpt_step0)
                print(f"New best step-0 model saved: {best_ckpt_step0} (val_acc={val_acc:.4f})")
                improve = True

            # Save best average model
            if avg_val_acc > best_val_acc_avg:
                best_val_acc_avg = avg_val_acc
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg,
                    "best_val_acc_step0": best_val_acc_step0,
                    "best_val_acc_avg": best_val_acc_avg
                }, best_ckpt_avg)
                print(f"New best avg model saved: {best_ckpt_avg} (avg_val_acc={avg_val_acc:.4f})")
                improve = True
            
            if improve:
                epochs_no_improve = 0
                improve = False
            else:
                epochs_no_improve += 1
                print(f"No improvement in val_accs for {epochs_no_improve} epochs.")

            if epochs_no_improve >= patience:
                early_stop = True

        # Final model save
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "cfg": cfg,
            "best_val_acc_step0": best_val_acc_step0,
            "best_val_acc_avg": best_val_acc_avg
        }, final_path)


if __name__ == "__main__":
    main()