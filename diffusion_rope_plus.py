# Code takes heavy inspiration from Andrej Karpathy's two implementations:
# nanochat: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
# "Let's build GPT" video: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import os
import sys
import time

import torch
import torch.nn as nn
from torch.nn import functional as F


def _getenv_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


def _getenv_float(name, default):
    value = os.getenv(name)
    return default if value is None else float(value)


def _getenv_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# hyperparameters
batch_size = _getenv_int(
    "TD_BATCH_SIZE", 64
)  # how many independent sequences will we process in parallel?
block_size = _getenv_int(
    "TD_BLOCK_SIZE", 256
)  # what is the maximum context length for predictions?
max_iters = _getenv_int("TD_MAX_ITERS", 10000)
eval_interval = _getenv_int("TD_EVAL_INTERVAL", 500)
learning_rate = _getenv_float("TD_LEARNING_RATE", 3e-4)
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
eval_iters = _getenv_int("TD_EVAL_ITERS", 200)
n_embd = _getenv_int("TD_N_EMBD", 384)
n_head = _getenv_int("TD_N_HEAD", 6)
n_layer = _getenv_int("TD_N_LAYER", 6)
head_dim = n_embd // n_head
# ------------
seed = _getenv_int("TD_SEED", 1337)
torch.manual_seed(seed)

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
        # Per-head mask-aware bias coefficients: [m_i, m_j, m_i*m_j].
        self.mask_bias_raw = nn.Parameter(torch.zeros(n_head, 3))
        self.mask_bias_max = float(_getenv_float("TD_MASK_BIAS_MAX", 0.5))
        self.mask_bias_clamp = float(_getenv_float("TD_MASK_BIAS_CLAMP", 2.0))

    def forward(self, x, cos_sin, token_is_mask, bias_scale):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)

        cos, sin = cos_sin
        cos = cos[:, :T].to(q.dtype)
        sin = sin[:, :T].to(q.dtype)

        # Keep positional encoding as pure RoPE.
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, T, H, D) -> (B, H, T, D)

        # Add bounded, warm-started mask-aware bias outside the dot product.
        m = token_is_mask.to(q.dtype)
        m_i = m[:, None, :, None]  # (B,1,T,1)
        m_j = m[:, None, None, :]  # (B,1,1,T)
        coeff = self.mask_bias_max * torch.tanh(self.mask_bias_raw).to(q.dtype)
        coeff = coeff * float(bias_scale)
        g1 = coeff[:, 0].view(1, n_head, 1, 1)
        g2 = coeff[:, 1].view(1, n_head, 1, 1)
        g3 = coeff[:, 2].view(1, n_head, 1, 1)
        attn_bias = g1 * m_i + g2 * m_j + g3 * (m_i * m_j)
        attn_bias = torch.clamp(attn_bias, -self.mask_bias_clamp, self.mask_bias_clamp).to(q.dtype)

        # [NEW]: Set to false for bidirectional instead of causal self-attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)

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

    def forward(self, x, cos_sin, token_is_mask, bias_scale):
        x = x + self.attn(norm(x), cos_sin, token_is_mask, bias_scale)
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
        self.bias_warmup_steps = _getenv_int("TD_MASK_BIAS_WARMUP_STEPS", 800)
        self.bias_ramp_steps = _getenv_int("TD_MASK_BIAS_RAMP_STEPS", 1200)
        # Default to fully enabled bias for inference-only usage.
        self.current_step = self.bias_warmup_steps + self.bias_ramp_steps

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

    def set_step(self, step):
        self.current_step = int(step)

    def _mask_bias_scale(self):
        step = int(self.current_step)
        if step < self.bias_warmup_steps:
            return 0.0
        if self.bias_ramp_steps <= 0:
            return 1.0
        return min(1.0, (step - self.bias_warmup_steps) / float(self.bias_ramp_steps))

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.size()

        # Get embeddings
        x = self.token_emb(idx)  # (B, T, n_embd)
        x = norm(x)

        # Get rotary embeddings
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        token_is_mask = idx == mask_token_id
        bias_scale = self._mask_bias_scale()

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin, token_is_mask, bias_scale)
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
    model, max_new_tokens, prompt_len=16, temp=1.0, confidence_threshold=0.95, top_k=3
):
    all_tokens = data[:prompt_len].tolist()
    total_steps = 0

    # Generate one block at a time
    while len(all_tokens) - prompt_len < max_new_tokens:
        # How many tokens to generate this block
        block_len = min(240, prompt_len + max_new_tokens - len(all_tokens))

        # Initialize: last prompt_len tokens + masks
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)

        # Track which positions need decoding
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        # Iteratively decode
        while masked.any():
            total_steps += 1

            # Get predictions and confidences
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            # Decode high-confidence masked positions (or at least 1)
            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences, torch.tensor(-float("inf"))
                )
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            # Sample from top-k and update
            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(
                1, block_size
            )
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

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
    train_flag = "--train" in sys.argv or _getenv_bool("TD_TRAIN", False)
    weights_path = os.getenv("TD_WEIGHTS_PATH", "weights/diffusion_rope_plus.pt")
    load_weights = _getenv_bool("TD_LOAD_WEIGHTS", True)
    save_weights = _getenv_bool("TD_SAVE_WEIGHTS", True)
    sample_during_train = _getenv_bool("TD_SAMPLE_DURING_TRAIN", True)
    run_generate = _getenv_bool("TD_RUN_GENERATE", True)
    train_sample_tokens = _getenv_int("TD_TRAIN_SAMPLE_TOKENS", 240)
    final_gen_tokens = _getenv_int("TD_FINAL_GEN_TOKENS", 2000)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    model = Model()
    m = model.to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # Load weights if they exist and train flag not set
    if load_weights and os.path.exists(weights_path) and not train_flag:
        print(f"Loading weights from {weights_path}")
        m.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Training from scratch")

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        best_val = float("inf")

        start = time.time()
        for iter in range(max_iters):
            model.set_step(iter)
            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                train_loss = losses["train"].item()
                val_loss = losses["val"].item()
                best_val = min(best_val, val_loss)
                print(
                    f"step {iter}: train loss {train_loss:.4f},"
                    f"val loss {val_loss:.4f}, time {time.time() - start:.2f} seconds"
                )
                print(
                    f"METRIC step={iter} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                )
                # Generate a sample
                if sample_during_train:
                    sample = generate(m, max_new_tokens=train_sample_tokens)
                    print(f"Sample:\n{sample}\n")

            # sample a batch of data
            xb, yb, mb = get_batch("train")

            # evaluate the loss
            logits, loss = model(xb, yb, mb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Save the model weights
        print(f"Total training time: {time.time() - start:.2f} seconds")
        print(f"METRIC best_val_loss={best_val:.6f}")
        if save_weights:
            print(f"Saving weights to {weights_path}")
            torch.save(m.state_dict(), weights_path)

    # generate from the model
    if run_generate:
        model.set_step(model.bias_warmup_steps + model.bias_ramp_steps)
        start = time.time()
        output = generate(
            m, max_new_tokens=final_gen_tokens, temp=0.8, confidence_threshold=0.95, top_k=2
        )
        print(f"Total generation time: {time.time() - start:.2f} seconds")
        print(f"\nOutput:\n{output}")
