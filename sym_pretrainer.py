import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from tqdm import tqdm
import os
import numpy as np
import shutil
from typing import Optional
from sym import make_key_padding_mask


def create_masked_positions(T: int, mask_prob: float, ensure_min_mask: int = 1):
    """
    Create a 1D boolean mask of length T where approximately mask_prob fraction are True.
    We'll ensure at least `ensure_min_mask` positions are masked to avoid degenerate cases.
    Returns shape (T,) numpy bool array.
    """
    if mask_prob <= 0.0:
        return np.zeros((T,), dtype=bool)
    if mask_prob >= 1.0:
        return np.ones((T,), dtype=bool)

    # target number of masked tokens
    n_mask = max(ensure_min_mask, int(round(mask_prob * T)))
    all_idx = np.arange(T)
    masked_idx = np.random.choice(all_idx, size=n_mask, replace=False)
    mask = np.zeros((T,), dtype=bool)
    mask[masked_idx] = True
    return mask



# -----------------------------
# Masked loss for pretraining
# -----------------------------
def sequence_multi_step_masked_loss(
    logits: torch.Tensor,        # (B, T, pred_steps, C)
    targets: torch.Tensor,       # (B, T)
    mask: torch.BoolTensor,      # (B, T) True = position to predict (masked)
    lengths: Optional[torch.Tensor] = None,
    step_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multi-step cross-entropy but computed only on masked positions.
    `mask` marks positions we want the network to predict (True = include in loss).
    Handles shifting for future pred_steps: for step k we consider mask[:, k:].
    """
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

        step_logits = logits[:, :-k or None, k, :]   # (B, T-k, C)
        step_targets = targets[:, k:]                # (B, T-k)
        step_mask = mask[:, k:]                      # (B, T-k)

        # If lengths given, make mask false beyond lengths
        if lengths is not None:
            # valid positions according to lengths (B, T-k)
            lengths_shifted = (lengths - k).clamp_min(0)
            padding_mask = make_key_padding_mask(lengths_shifted, T - k)  # (B, T-k) True for pad
            valid_pos = ~padding_mask
            step_mask = step_mask & valid_pos

        # If nothing to compute for this step, skip
        if step_mask.sum().item() == 0:
            continue

        # Compute per-position cross entropy (B*(T-k) flattened)
        step_loss = F.cross_entropy(
            step_logits.reshape(-1, C),
            step_targets.reshape(-1),
            reduction="none"
        ).reshape(B, T - k)

        # Keep only masked positions
        masked_loss = (step_loss * step_mask.float()).sum()
        denom = step_mask.float().sum().clamp_min(1.0)
        step_loss_val = masked_loss / denom

        w = step_weights[k]
        total_loss += w * step_loss_val
        total_weight += w

    if total_weight == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / total_weight


# -----------------------------
# Pretraining loops
# -----------------------------
def pretrain_token_phase(
    model,
    optimizer,
    train_gen,
    device,
    steps,                    # total steps to run for this phase
    mask_start=0.05,
    mask_end=0.5,
    lambda_vq=0.0,
    log_interval=100,
    save_dir=None,
    model_name="model",
    global_step_offset=0,
    checkpoint_interval=5000,
):
    """
    Token-only pretraining phase: model receives no audio features (we supply zero preencoded audio)
    but DOES receive aux_steps. We mask input tokens progressively from mask_start -> mask_end.
    Only masked positions are used in the loss.
    """
    model.train()
    pbar = tqdm(range(steps), desc="Pretrain (tokens only)", leave=False)
    total_loss = 0.0

    for step in pbar:
        ns2, ns0, ns1 = next(train_gen)  # audio, aux, targets
        ns2, ns0, ns1 = np.array(ns2), np.array(ns0), np.array(ns1)

        # For token-only phase, we still require aux. audio can be ignored.
        aux = torch.as_tensor(ns0, dtype=torch.float32).unsqueeze(0).to(device)

        if ns1.ndim == 2:
            target_ids_np = ns1.argmax(axis=-1).astype(np.int64)
        else:
            target_ids_np = ns1.astype(np.int64)

        T = int(target_ids_np.shape[0])
        # create input tokens (teacher-forced shifted inputs)
        token_in = np.full((1, T), fill_value=model.bos_id, dtype=np.int64)
        if T > 1:
            token_in[0, 1:] = target_ids_np[:-1]
        token_in_t = torch.as_tensor(token_in, dtype=torch.long).to(device)
        target_t = torch.as_tensor(target_ids_np, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([T], device=device)

        # Determine mask probability for this step (linear schedule)
        progress = (step + global_step_offset) / max(1, steps + global_step_offset)
        mask_prob = mask_start + (mask_end - mask_start) * progress
        mask_np = create_masked_positions(T, mask_prob)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(device)  # (1, T)

        # Replace masked positions in token inputs with mask token (we reuse bos_id as mask token)
        token_in_masked = token_in.copy()
        token_in_masked[0, mask_np] = model.bos_id
        token_in_masked_t = torch.as_tensor(token_in_masked, dtype=torch.long).to(device)

        # Prepare zero audio embeddings as preencoded audio so forward path works unchanged
        # audio_emb shape must be (B, T, d_model)
        d_model = model.cfg.d_model
        audio_emb_zero = torch.zeros((1, T, d_model), dtype=torch.float32, device=device)
        preencoded_audio = (audio_emb_zero, torch.tensor(0.0, device=device))

        logits, enc_loss = model(
            None,
            aux,
            token_in_masked_t,
            lengths=lengths,
            preencoded_audio=preencoded_audio
        )

        # build uniform step weights or use scheduler if desired (here uniform)
        step_weights = torch.ones(model.pred_steps, device=device)
        loss = sequence_multi_step_masked_loss(logits, target_t, mask_t, lengths=lengths, step_weights=step_weights)
        # include enc_loss scaled by lambda_vq (likely zero here)
        loss = loss + lambda_vq * enc_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            avg = total_loss / log_interval
            pbar.set_postfix(loss=f"{avg:.4f}", mask_prob=f"{mask_prob:.3f}")
            total_loss = 0.0

        global_step = global_step_offset + step
        # periodic checkpointing (same pattern as before)
        if save_dir and ((global_step + 1) % checkpoint_interval == 0 or (step + 1) == steps):
            latest_path = os.path.join(save_dir, f"{model_name}_pretoken_latest.pt")

            # remove older checkpoint(s) for this phase
            for f in os.listdir(save_dir):
                if f.startswith(f"{model_name}_pretoken_checkpoint_step") and f.endswith(".pt"):
                    os.remove(os.path.join(save_dir, f))

            ckpt_path = os.path.join(save_dir, f"{model_name}_pretoken_checkpoint_step{global_step+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "phase": "pretoken",
                "global_step": global_step + 1
            }, ckpt_path)
            shutil.copy(ckpt_path, latest_path)
            os.remove(ckpt_path)


def pretrain_audio_phase(
    model,
    optimizer,
    train_gen,
    device,
    steps,
    mask_start=0.05,
    mask_end=0.6,
    lambda_vq=0.25,
    log_interval=100,
    save_dir=None,
    model_name="model",
    global_step_offset=0,
    checkpoint_interval=5000,
):
    """
    Same as token phase but encodes audio via model.encode_audio and includes enc_loss in objective.
    """
    model.train()
    pbar = tqdm(range(steps), desc="Pretrain (audio introduced)", leave=False)
    total_loss = 0.0

    for step in pbar:
        ns2, ns0, ns1 = next(train_gen)  # audio, aux, targets
        ns2, ns0, ns1 = np.array(ns2), np.array(ns0), np.array(ns1)

        audio = torch.as_tensor(ns2, dtype=torch.float32).unsqueeze(0).to(device)
        aux = torch.as_tensor(ns0, dtype=torch.float32).unsqueeze(0).to(device)

        if ns1.ndim == 2:
            target_ids_np = ns1.argmax(axis=-1).astype(np.int64)
        else:
            target_ids_np = ns1.astype(np.int64)

        T = int(target_ids_np.shape[0])
        token_in = np.full((1, T), fill_value=model.bos_id, dtype=np.int64)
        if T > 1:
            token_in[0, 1:] = target_ids_np[:-1]
        token_in_t = torch.as_tensor(token_in, dtype=torch.long).to(device)
        target_t = torch.as_tensor(target_ids_np, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([audio.shape[1]], device=device)

        # mask progress schedule
        progress = (step + global_step_offset) / max(1, steps + global_step_offset)
        mask_prob = mask_start + (mask_end - mask_start) * progress
        mask_np = create_masked_positions(T, mask_prob)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(device)

        # Replace masked positions in token inputs with mask token (reuse bos_id)
        token_in_masked = token_in.copy()
        token_in_masked[0, mask_np] = model.bos_id
        token_in_masked_t = torch.as_tensor(token_in_masked, dtype=torch.long).to(device)

        # Pre-encode audio using model.encode_audio (so we get vq_loss)
        audio_emb, enc_loss = model.encode_audio(audio, aux)

        logits, _ = model(
            None,
            aux,
            token_in_masked_t,
            lengths=lengths,
            preencoded_audio=(audio_emb, enc_loss)
        )

        step_weights = torch.ones(model.pred_steps, device=device)
        loss = sequence_multi_step_masked_loss(logits, target_t, mask_t, lengths=lengths, step_weights=step_weights)
        loss = loss + lambda_vq * enc_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            avg = total_loss / log_interval
            pbar.set_postfix(loss=f"{avg:.4f}", mask_prob=f"{mask_prob:.3f}")
            total_loss = 0.0

        global_step = global_step_offset + step
        # periodic checkpointing for audio pretrain phase
        if save_dir and ((global_step + 1) % checkpoint_interval == 0 or (step + 1) == steps):
            latest_path = os.path.join(save_dir, f"{model_name}_preaudio_latest.pt")

            for f in os.listdir(save_dir):
                if f.startswith(f"{model_name}_preaudio_checkpoint_step") and f.endswith(".pt"):
                    os.remove(os.path.join(save_dir, f))

            ckpt_path = os.path.join(save_dir, f"{model_name}_preaudio_checkpoint_step{global_step+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "phase": "preaudio",
                "global_step": global_step + 1
            }, ckpt_path)
            shutil.copy(ckpt_path, latest_path)
            os.remove(ckpt_path)