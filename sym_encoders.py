import torch
import torch.nn as nn
from util import *
from typing import Tuple
from vector_quantize_pytorch import ResidualVQ

#-----------------------------
# Conv encoder for per-step (S,F,C)
# -----------------------------

class StepConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        hidden_dim: int = 256,
        do_vq: bool = False,
        vq_dim: int = 128,
        num_codes: int = 512,
        conv_dim: int = 32,
    ):
        super().__init__()

        # CNN frontend (unchanged — this is correct)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, kernel_size=(7,3), padding=(3,1)),
            nn.GELU(),
            nn.Conv2d(conv_dim, 2*conv_dim, kernel_size=3, stride=(1,2), padding=1),
            nn.GELU(),
            nn.Conv2d(2*conv_dim, vq_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # 2D positional embeddings (moved before VQ)
        self.pos_emb_time = nn.Embedding(512, vq_dim)
        self.pos_emb_freq = nn.Embedding(512, vq_dim)
        self.pos_ln = nn.LayerNorm(vq_dim)  # normalize after pos injection

        self.do_vq = do_vq
        if self.do_vq:
            # VQ now sees position-aware patches
            self.vq = ResidualVQ(
                dim=vq_dim,
                num_quantizers=4,
                codebook_size=num_codes,
                decay=0.99,
                commitment_weight=0.25,
                threshold_ema_dead_code=2,
                kmeans_init=True,
                use_cosine_sim=True,
                shared_codebook=False,
            )

        # Transformer over quantized, position-aware patches
        ffn_dim = max(hidden_dim, vq_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vq_dim, nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.seq_pool = nn.Linear(vq_dim, 1)

        # CNN bypass path (regularized)
        self.cnn_ff = nn.Sequential(
            nn.Linear(vq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vq_dim),
            nn.Dropout(0.15),
        )

        # Per-channel gate (was scalar, now vq_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * vq_dim, vq_dim),
            nn.GELU(),
            nn.Linear(vq_dim, vq_dim),
        )

        self.proj = nn.Linear(vq_dim, d_model)
        self.layer_norm = nn.LayerNorm(vq_dim)

    def forward(self, x: torch.Tensor):
        B, T, S, Freq, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B*T, C, S, Freq)

        y = self.cnn(x)
        _, C_out, S_new, F_new = y.shape

        # Build and inject positional embeddings BEFORE VQ
        pos_time = torch.arange(S_new, device=x.device).unsqueeze(1).expand(S_new, F_new)
        pos_freq = torch.arange(F_new, device=x.device).unsqueeze(0).expand(S_new, F_new)
        pos_emb = (
            self.pos_emb_time(pos_time) + self.pos_emb_freq(pos_freq)
        ).view(1, S_new * F_new, C_out)

        y_seq = y.permute(0, 2, 3, 1).contiguous().view(B*T, S_new*F_new, C_out)
        y_seq = self.layer_norm(y_seq)
 

        # VQ on position-aware patches
        if self.do_vq:
            y_q, _, vq_loss = self.vq(y_seq)
            vq_loss = vq_loss.mean()
        else:
            y_q = y_seq
            vq_loss = 0
        y_q = y_q + pos_emb

        # Transformer attends over position-encoded quantized patches
        y_trans = self.transformer(y_q)

        # Attention pooling over transformer output
        attn_weights = torch.softmax(self.seq_pool(y_trans).squeeze(-1), dim=-1)
        y_trans_pooled = torch.bmm(attn_weights.unsqueeze(1), y_trans).squeeze(1)

        # CNN bypass: global spectral summary (regularized)
        y_cnn_pool = y.mean(dim=(2, 3))
        y_cnn_ff = self.cnn_ff(y_cnn_pool)

        # Per-channel gate fusion
        gate_in = torch.cat([y_trans_pooled, y_cnn_ff], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_in))          # (B*T, vq_dim)
        y = (1 - gate) * y_trans_pooled + gate * y_cnn_ff

        y = self.proj(y).view(B, T, -1)
        return y, vq_loss



class TemporalResidualBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        residual = x
        y = x.permute(0, 2, 1)  # (B, D, L)
        y = self.conv1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = y.permute(0, 2, 1)  # (B, L, D)
        return self.norm(residual + y)


class SimpleStepConvEncoder(nn.Module):
    """
    Conv2D frontend -> temporal residual blocks -> Residual VQ (Library version).
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        conv2d_channels: int = 64,
        temporal_layers: int = 4,
        temporal_kernel: int = 3,
        rvq_codebooks: int = 4,
        rvq_codebook_size: int = 512,
        use_segment_transformer: bool = False,
        segment_transformer_heads: int = 4,
        segment_transformer_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Conv2D frontend
        self.frontend = nn.Sequential(
            nn.Conv2d(in_channels, conv2d_channels, kernel_size=(7, 3), padding=(3, 1)),
            nn.GELU(),
            nn.Conv2d(conv2d_channels, conv2d_channels * 2, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(conv2d_channels * 2, d_model, kernel_size=3, stride=(1, 2), padding=1),
            nn.GELU(),
        )

        # temporal residual blocks
        self.temporal_blocks = nn.ModuleList([
            TemporalResidualBlock(d_model, kernel_size=temporal_kernel, dilation=2 ** (i % 4), dropout=dropout)
            for i in range(temporal_layers)
        ])

        # optional segment transformer
        self.use_segment_transformer = use_segment_transformer
        if use_segment_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=segment_transformer_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            )
            self.segment_transformer = nn.TransformerEncoder(enc_layer, num_layers=segment_transformer_layers)

        # Swapped: vector-quantize-pytorch ResidualVQ
        # kmeans_init=True is highly recommended to prevent the "derailing" you mentioned
        self.rvq = ResidualVQ(
            dim = d_model,
            num_quantizers = rvq_codebooks,
            codebook_size = rvq_codebook_size,
            decay = 0.99,
            commitment_weight = 0.25,
            threshold_ema_dead_code = 2,
            kmeans_init = True,   
            use_cosine_sim = True
        )

        # projection for pooled output
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, S, Freq, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B * T, C, S, Freq)
        
        y = self.frontend(x)  # (B*T, d_model, S', F')
        y = y.mean(dim=-1)     # (B*T, d_model, S_new)
        y = y.permute(0, 2, 1).contiguous()  # (B*T, S_new, d_model)

        # temporal residual blocks
        for blk in self.temporal_blocks:
            y = blk(y)

        # optional local transformer
        if self.use_segment_transformer:
            y = self.segment_transformer(y)

        # Library Call: ResidualVQ
        # Returns: (quantized, indices, multi-layer loss)
        y_q, _, vq_loss = self.rvq(y)

        vq_loss = vq_loss.mean()

        # Sum the vq_loss if the library returns it per-layer (depends on version/settings)
        # Usually it returns a single scalar representing the sum of all quantizer losses.
        vq_loss = vq_loss.mean() 

        pooled = y_q.mean(dim=1)  # Mean pool across the temporal dimension S_new
        out = self.proj(pooled).view(B, T, -1)

        return out, vq_loss
    

class DDCStepConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        hidden_dim: int = 256,
        do_vq: bool = False,
        vq_dim: int = 128,
        num_codes: int = 512,
        conv_dim: int = 32,
    ):
        super().__init__()

        # Modified CNN frontend based on your specific steps:
        # 1. (7,3) pass with no padding
        # 2. Max pooling with a (1,3) kernel and (1,3) stride
        # 3. (3,3) kernel with no padding
        # 4. Max pooling layer at the bottom with a (1,3) kernel and (1,3) stride
        self.cnn = nn.Sequential(
            # First Conv Pass
            nn.Conv2d(in_channels, conv_dim, kernel_size=(7, 3), padding=0),
            nn.GELU(),
            # First MaxPool (Frequency shrunk by a third)
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Second Conv Pass
            nn.Conv2d(conv_dim, vq_dim, kernel_size=(3, 3), padding=0),
            nn.GELU(),
            # Final MaxPool at the bottom (Frequency shrunk by a third again)
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        # 2D positional embeddings (moved before VQ)
        self.pos_emb_time = nn.Embedding(512, vq_dim)
        self.pos_emb_freq = nn.Embedding(512, vq_dim)
        self.pos_ln = nn.LayerNorm(vq_dim)  # normalize after pos injection

        self.do_vq = do_vq
        if self.do_vq:
            # VQ now sees position-aware patches
            self.vq = ResidualVQ(
                dim=vq_dim,
                num_quantizers=4,
                codebook_size=num_codes,
                decay=0.99,
                commitment_weight=0.25,
                threshold_ema_dead_code=2,
                kmeans_init=True,
                use_cosine_sim=True,
                shared_codebook=False,
            )

        # Transformer over quantized, position-aware patches
        ffn_dim = max(hidden_dim, vq_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vq_dim, nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.seq_pool = nn.Linear(vq_dim, 1)

        # CNN bypass path (regularized)
        self.cnn_ff = nn.Sequential(
            nn.Linear(vq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vq_dim),
            nn.Dropout(0.15),
        )

        # Per-channel gate (was scalar, now vq_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * vq_dim, vq_dim),
            nn.GELU(),
            nn.Linear(vq_dim, vq_dim),
        )

        self.proj = nn.Linear(vq_dim, d_model)
        self.layer_norm = nn.LayerNorm(vq_dim)

    def forward(self, x: torch.Tensor):
        B, T, S, Freq, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B*T, C, S, Freq)

        y = self.cnn(x)
        _, C_out, S_new, F_new = y.shape

        # Build and inject positional embeddings BEFORE VQ
        pos_time = torch.arange(S_new, device=x.device).unsqueeze(1).expand(S_new, F_new)
        pos_freq = torch.arange(F_new, device=x.device).unsqueeze(0).expand(S_new, F_new)
        pos_emb = (
            self.pos_emb_time(pos_time) + self.pos_emb_freq(pos_freq)
        ).view(1, S_new * F_new, C_out)

        y_seq = y.permute(0, 2, 3, 1).contiguous().view(B*T, S_new*F_new, C_out)
        y_seq = self.layer_norm(y_seq)
 

        # VQ on position-aware patches
        if self.do_vq:
            y_q, _, vq_loss = self.vq(y_seq)
            vq_loss = vq_loss.mean()
        else:
            y_q = y_seq
            vq_loss = 0
        y_q = y_q + pos_emb

        # Transformer attends over position-encoded quantized patches
        y_trans = self.transformer(y_q)

        # Attention pooling over transformer output
        attn_weights = torch.softmax(self.seq_pool(y_trans).squeeze(-1), dim=-1)
        y_trans_pooled = torch.bmm(attn_weights.unsqueeze(1), y_trans).squeeze(1)

        # CNN bypass: global spectral summary (regularized)
        y_cnn_pool = y.mean(dim=(2, 3))
        y_cnn_ff = self.cnn_ff(y_cnn_pool)

        # Per-channel gate fusion
        gate_in = torch.cat([y_trans_pooled, y_cnn_ff], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_in))          # (B*T, vq_dim)
        y = (1 - gate) * y_trans_pooled + gate * y_cnn_ff

        y = self.proj(y).view(B, T, -1)
        return y, vq_loss
    


class BasicEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        conv_dim: int = 32,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, kernel_size=(7, 3), stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=(3, 3), stride=(1,3) , padding=(1,0)),
            nn.GELU(),
            nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=(3, 3), stride=1 , padding=0),
            nn.GELU(),
            nn.Conv2d(conv_dim*2, conv_dim*2, kernel_size=(3, 3), stride=(1,3) , padding=(1,0)),
            nn.GELU(),
            nn.Conv2d(conv_dim*2, conv_dim*2, kernel_size=(13, 3), padding=(0,1)),
            nn.GELU(),
        )

        self.norm = nn.RMSNorm(d_model)

        self.pool = nn.Linear(16 * conv_dim, d_model)

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        B, T, S, Freq, C = x.shape
        # Flatten time into batch: [B * T, C, S, Freq]
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B * T, C, S, Freq)

        # CNN forward pass
        y = self.cnn(x)
        _, C_out, S_new, F_new = y.shape  # e.g., [B * T, conv_dim, S_new, F_new]
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(B * T, -1)
        y = self.norm(self.pool(y))

        y_out = self.proj(y).view(B, T, -1)
        return y_out, 0
    


class NewStepConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        hidden_dim: int = 256,
        do_vq: bool = False,
        vq_dim: int = 128,
        num_codes: int = 512,
        conv_dim: int = 32,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, kernel_size=(7, 3), stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=(3, 3), stride=(1,3) , padding=(1,0)),
            nn.GELU(),
            nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=(3, 3), stride=1 , padding=0),
            nn.GELU(),
            nn.Conv2d(conv_dim*2, vq_dim, kernel_size=(3, 3), stride=(1,3) , padding=(1,0)),
            nn.GELU(),
        )

        # 2D positional embeddings (moved before VQ)
        self.pos_emb_time = nn.Embedding(512, vq_dim)
        self.pos_emb_freq = nn.Embedding(512, vq_dim)
        self.pos_ln = nn.LayerNorm(vq_dim)  # normalize after pos injection

        self.do_vq = do_vq
        if self.do_vq:
            # VQ now sees position-aware patches
            self.vq = ResidualVQ(
                dim=vq_dim,
                num_quantizers=4,
                codebook_size=num_codes,
                decay=0.99,
                commitment_weight=0.25,
                threshold_ema_dead_code=2,
                kmeans_init=True,
                use_cosine_sim=True,
                shared_codebook=False,
            )

        # Transformer over quantized, position-aware patches
        ffn_dim = max(hidden_dim, vq_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vq_dim, nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.seq_pool = nn.Linear(vq_dim, 1)

        # CNN bypass path (regularized)
        self.cnn_ff = nn.Sequential(
            nn.Linear(vq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vq_dim),
            nn.Dropout(0.15),
        )

        # Per-channel gate (was scalar, now vq_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * vq_dim, vq_dim),
            nn.GELU(),
            nn.Linear(vq_dim, vq_dim),
        )

        self.proj = nn.Linear(vq_dim, d_model)
        self.layer_norm = nn.LayerNorm(vq_dim)

    def forward(self, x: torch.Tensor):
        B, T, S, Freq, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B*T, C, S, Freq)

        y = self.cnn(x)
        _, C_out, S_new, F_new = y.shape

        # Build and inject positional embeddings BEFORE VQ
        pos_time = torch.arange(S_new, device=x.device).unsqueeze(1).expand(S_new, F_new)
        pos_freq = torch.arange(F_new, device=x.device).unsqueeze(0).expand(S_new, F_new)
        pos_emb = (
            self.pos_emb_time(pos_time) + self.pos_emb_freq(pos_freq)
        ).view(1, S_new * F_new, C_out)

        y_seq = y.permute(0, 2, 3, 1).contiguous().view(B*T, S_new*F_new, C_out)
        y_seq = self.layer_norm(y_seq)
 

        # VQ on position-aware patches
        if self.do_vq:
            y_q, _, vq_loss = self.vq(y_seq)
            vq_loss = vq_loss.mean()
        else:
            y_q = y_seq
            vq_loss = 0
        y_q = y_q + pos_emb

        # Transformer attends over position-encoded quantized patches
        y_trans = self.transformer(y_q)

        # Attention pooling over transformer output
        attn_weights = torch.softmax(self.seq_pool(y_trans).squeeze(-1), dim=-1)
        y_trans_pooled = torch.bmm(attn_weights.unsqueeze(1), y_trans).squeeze(1)

        # CNN bypass: global spectral summary (regularized)
        y_cnn_pool = y.mean(dim=(2, 3))
        y_cnn_ff = self.cnn_ff(y_cnn_pool)

        # Per-channel gate fusion
        gate_in = torch.cat([y_trans_pooled, y_cnn_ff], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_in))          # (B*T, vq_dim)
        y = (1 - gate) * y_trans_pooled + gate * y_cnn_ff

        y = self.proj(y).view(B, T, -1)
        return y, vq_loss
    

class EffortEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        conv_dim: int = 32,
        nhead: int = 4,
    ):
        super().__init__()

        # First 4 conv layers (preserving spatial dimensions to yield ~ [B*T, conv_dim*2, 13, 8])
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, kernel_size=(7, 3), stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=(3, 3), stride=(1, 3), padding=(1, 0)),
            nn.GELU(),
            nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=(3, 3), stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(conv_dim * 2, conv_dim * 2, kernel_size=(3, 3), stride=(1, 3), padding=(1, 0)),
            nn.GELU(),
        )
        
        # Inferred output channel from the 4th conv layer
        c_in = conv_dim * 2  

        # --- Embeddings to d_model ---
        # Time path: input shape is [B*T, 13, 8*C] -> project 8*C to d_model
        self.embed_time = nn.Linear(8 * c_in, d_model)
        # Freq path: input shape is [B*T, 8, 13*C] -> project 13*C to d_model
        self.embed_freq = nn.Linear(13 * c_in, d_model)

        # --- Learned Positional Embeddings ---
        self.pos_time = nn.Parameter(torch.zeros(1, 13, d_model))
        self.pos_freq = nn.Parameter(torch.zeros(1, 8, d_model))

        # --- Attention Layers ---
        # Batch_first=True makes handling the B*T dimension clean
        self.time_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.freq_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # Cross attention: Queries from Frame (Time), Keys/Values from Freq
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # Final normalization and projection
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        B, T, S, Freq, C = x.shape
        # Flatten time into batch: [B * T, C, S, Freq]
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B * T, C, S, Freq)

        # 1. CNN forward pass (First 4 layers)
        # Output shape: [B * T, C_out, 13, 8]
        y = self.cnn(x)
        BT, C_out, H, W = y.shape  # H = 13 (Time), W = 8 (Freq)

        # 2. Path A: Time Frame Attention
        # Permute to [B*T, 13, 8, C_out] -> reshape to [B*T, 13, 8 * C_out]
        y_time = y.permute(0, 2, 3, 1).reshape(BT, H, W * C_out)
        y_time = self.embed_time(y_time) + self.pos_time
        # Self-attend across the 13 time frames
        time_attn_out, _ = self.time_self_attn(y_time, y_time, y_time)

        # 3. Path B: Frequency Attention
        # Permute to [B*T, 8, 13, C_out] -> reshape to [B*T, 8, 13 * C_out]
        y_freq = y.permute(0, 3, 2, 1).reshape(BT, W, H * C_out)
        y_freq = self.embed_freq(y_freq) + self.pos_freq
        # Self-attend across the 8 frequency bins
        freq_attn_out, _ = self.freq_self_attn(y_freq, y_freq, y_freq)

        # 4. Cross Attention
        # Cross attend Freq attention (K, V) to Frame attention (Q)
        # Output shape matches Query shape: [B*T, 13, d_model]
        joint_features, _ = self.cross_attn(query=time_attn_out, key=freq_attn_out, value=freq_attn_out)

        # 5. Global Mean Pooling across the remaining sequence dimension (13)
        # [B*T, 13, d_model] -> [B*T, d_model]
        y_pooled = joint_features.mean(dim=1)

        # 6. Final normalization and linear layer
        y_out = self.norm(y_pooled)
        y_out = self.proj(y_out).view(B, T, -1)

        return y_out, 0