# Code takes heavy inspiration from Andrej Karpathy's two implementations:
# nanochat: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
# "Let's build GPT" video: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import os
import sys
import time
import csv

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head
# ------------
torch.manual_seed(1337)

# Load data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# All the unique characters that occur in this text
chars = sorted(list(set(text)))
chars = ["_"] + chars  # [NEW]: Add underscore (doesn't appear in text)
vocab_size = len(chars)
# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
mask_token_id = stoi["_"]  # [NEW]: Set mask token to underscore


# encoder: take a string, output a list of integers
def encode(s):
    return [stoi[ch] for ch in s]


# decoder: take a list of integers, output a string
def decode(l):
    return "".join([itos[n] for n in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# [NEW]: Modify get batch to do masking
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = x.clone()  # original tokens

    # Mask tokens with random probability per sample
    mask_probs = torch.rand(batch_size, 1)
    mask = torch.rand(batch_size, block_size) < mask_probs
    x[mask] = mask_token_id

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, T, H, D) -> (B, H, T, D)

        # [NEW]: Set to false for bidirectional instead of causal self-attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Re-assemble the heads and project back
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Rotary embeddings
        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])

        # Output head to predict denoised tokens
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims
        return cos, sin

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.size()

        # Get embeddings
        x = self.token_emb(idx)  # (B, T, n_embd)
        x = norm(x)

        # Get rotary embeddings
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)

        # Predict denoised tokens
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            # [NEW]: Only compute loss on masked tokens if mask is provided
            if mask is not None:
                mask_flat = mask.view(B * T)
                loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                loss = (loss * mask_flat).sum() / mask_flat.sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# [NEW]: Change next-token-prediction to confidence-based parallel decoding
@torch.no_grad()
def generate(
    model,
    max_new_tokens,
    prompt_len=16,
    temp=1.0,
    confidence_threshold=0.95,
    top_k=3,
    draft_threshold=0.70,
    confirm_threshold=None,
    replace_margin=0.0,
    target_chunk_len=240,
):
    if confirm_threshold is None:
        confirm_threshold = confidence_threshold
    if not (0.0 <= draft_threshold <= 1.0 and 0.0 <= confirm_threshold <= 1.0):
        raise ValueError("draft_threshold and confirm_threshold must be in [0, 1]")
    if draft_threshold > confirm_threshold:
        raise ValueError("draft_threshold should be <= confirm_threshold")
    if target_chunk_len <= 0:
        raise ValueError("target_chunk_len must be > 0")

    all_tokens = data[:prompt_len].tolist()
    total_steps = 0

    # Generate one block at a time
    while len(all_tokens) - prompt_len < max_new_tokens:
        # How many tokens to generate this block
        block_len = min(target_chunk_len, prompt_len + max_new_tokens - len(all_tokens))

        # Initialize: last prompt_len tokens + masks
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)

        # Track decode states inside target region:
        # pending: unseen positions, draft: provisional positions, confirmed: final positions.
        target_mask = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        target_mask[0, prompt_len : prompt_len + block_len] = True
        pending = target_mask.clone()
        draft = torch.zeros_like(target_mask)
        confirmed = torch.zeros_like(target_mask)
        draft_conf = torch.full((1, block_size), -float("inf"), device=device)

        # Iteratively decode with two thresholds:
        # 1) low threshold for draft fill, 2) high threshold for final confirmation.
        while confirmed.sum().item() < block_len:
            total_steps += 1

            # Get predictions and confidences
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)
            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(
                1, block_size
            )
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            # Confirm tokens whose current confidence is already high enough.
            confirm_candidates = (pending | draft) & (confidences >= confirm_threshold)
            confirm_candidates = confirm_candidates & target_mask
            if confirm_candidates.any():
                x = torch.where(confirm_candidates, sampled_tokens, x)
                confirmed = confirmed | confirm_candidates
                pending = pending & ~confirm_candidates
                draft = draft & ~confirm_candidates

            # Fill pending positions as drafts using the low threshold.
            new_draft = pending & (confidences >= draft_threshold)
            if new_draft.any():
                x = torch.where(new_draft, sampled_tokens, x)
                draft = draft | new_draft
                pending = pending & ~new_draft
                draft_conf = torch.where(new_draft, confidences, draft_conf)

            # For draft positions, keep the token but allow replacement if confidence improves.
            replace_candidates = (
                draft
                & (confidences >= draft_threshold)
                & (confidences > (draft_conf + replace_margin))
            )
            if replace_candidates.any():
                x = torch.where(replace_candidates, sampled_tokens, x)
                draft_conf = torch.where(replace_candidates, confidences, draft_conf)

            progressed = (
                confirm_candidates.any() or new_draft.any() or replace_candidates.any()
            )
            if progressed:
                continue

            # Fallback:
            # - if there are pending positions, force the best pending position into draft.
            # - if only draft remains and no improvement happened, finalize drafts as-is.
            if pending.any():
                pending_conf = torch.where(
                    pending, confidences, torch.full_like(confidences, -float("inf"))
                )
                forced = torch.zeros_like(pending)
                forced.view(-1)[pending_conf.argmax()] = True
                x = torch.where(forced, sampled_tokens, x)
                draft = draft | forced
                pending = pending & ~forced
                draft_conf = torch.where(forced, confidences, draft_conf)
            elif draft.any():
                # No pending token left and drafts no longer improve: accept drafts.
                confirmed = confirmed | draft
                draft = torch.zeros_like(draft)

        # Extract and append generated tokens
        all_tokens.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    tokens_generated = len(all_tokens) - prompt_len
    print(f"Total steps: {total_steps} for {tokens_generated} tokens")
    print(f"Avg decoded per step: {tokens_generated / total_steps:.2f}")
    return decode(all_tokens)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(split)
            _, loss = model(X, Y, M)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    train_flag = "--train" in sys.argv
    weights_path = "weights/diffusion.pt"
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    model = Model()
    m = model.to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # Load weights if they exist and train flag not set
    if os.path.exists(weights_path) and not train_flag:
        print(f"Loading weights from {weights_path}")
        m.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Training from scratch")

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join(
            "logs", f"diffusion_train_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        print(f"Training log: {log_path}")

        start = time.time()
        last_batch_loss = float("nan")
        with open(log_path, "w", newline="", encoding="utf-8") as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(
                ["step", "elapsed_sec", "train_loss", "val_loss", "batch_total_loss"]
            )

            for iter in range(max_iters):
                # every once in a while evaluate the loss on train and val sets
                if iter % eval_interval == 0 or iter == max_iters - 1:
                    losses = estimate_loss()
                    elapsed = time.time() - start
                    log_writer.writerow(
                        [
                            iter,
                            f"{elapsed:.4f}",
                            f"{losses['train']:.6f}",
                            f"{losses['val']:.6f}",
                            f"{last_batch_loss:.6f}",
                        ]
                    )
                    log_file.flush()
                    print(
                        f"step {iter}: train loss {losses['train']:.4f},"
                        f"val loss {losses['val']:.4f}, time {elapsed:.2f} seconds"
                    )
                    # Generate a sample
                    sample = generate(m, max_new_tokens=240)
                    print(f"Sample:\n{sample}\n")

                # sample a batch of data
                xb, yb, mb = get_batch("train")

                # evaluate the loss
                logits, loss = model(xb, yb, mb)
                last_batch_loss = loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        # Save the model weights
        print(f"Total training time: {time.time() - start:.2f} seconds")
        print(f"Saving weights to {weights_path}")
        torch.save(m.state_dict(), weights_path)

    # generate from the model
    start = time.time()
    output = generate(
        m,
        max_new_tokens=2000,
        temp=0.8,
        confidence_threshold=0.95,
        top_k=2,
        draft_threshold=0.70,
        confirm_threshold=0.85,
        target_chunk_len=240,
    )
    print(f"Total generation time: {time.time() - start:.2f} seconds")
    print(f"\nOutput:\n{output}")
