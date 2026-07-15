from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_self_pre: int = 6      # token self-attn layers before cross-attn
    n_self_audio_pre: int = 2 # audio self-attn layers before cross-attn
    n_self_post: int = 6     # token self-attn layers after cross-attn
    n_cross: int = 2         # number of cross-attend hops (new)
    dropout: float = 0.1
    max_len: int = 2048
    num_classes: int = 256   # number of classes (excludes BOS)
    aux_dim: int = 2         # (last, next) or 3 if you include diff
    VQ_dim: int = 128        # dimension of VQ embeddings
    VQ_codes: int = 512      # number of VQ codes
    conv_dim: int = 32
    pred_steps: int = 4  
    conv_layers: int = 4
    conv_heads: int = 4
    tok_dropout: float = 0.2    # warmup target; training starts at 0 and ramps up to this
    # --- motif detection ---
    n_motifs: int = 1024
    motif_window_sizes: tuple = (4, 6, 8)  # parallel causal window lengths
    motif_phrase_dim: int = 32             # per-token embedding dim inside phrase encoder
    motif_hidden_dim: int = 128       # detection-space dim (phrase vectors -> motif keys)
    motif_n_heads: int = 4            # heads for all four SA/CA blocks in MotifDetector
    motif_conf_threshold: float = 0.3
    motif_lambda: float = 0.01        # sparsity aux loss weight
    motif_mode: str = "cross"    # "cross" | "gate" | "none" (disables motif entirely)
    motif_recon_lambda: float = 0.1   # start conservative; entropy loss weight is 0.01
    # --- token stream VQ bottleneck ---
    latent_vq: bool = False
    tok_vq_codes: int = 512          # codebook size — keep small for regularisation
    tok_vq_num_quantizers: int = 1   # single stage: most restrictive bottleneck
    tok_vq_weight: float = 0.25      # commitment loss weight (matches audio encoder)