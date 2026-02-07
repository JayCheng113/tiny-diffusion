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


class MultiHeadAttention(nn.Module):
    def __init__(self, mask_gate_alpha=0.3):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.phi = nn.Parameter(torch.zeros(head_dim))
        self.mask_gate_alpha = float(mask_gate_alpha)

    def forward(self, x, pos_gate, token_is_mask):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_head, head_dim)
        v = self.c_v(x).view(B, T, n_head, head_dim)

        a = torch.where(token_is_mask, self.mask_gate_alpha, 1.0)
        a = a.to(q.dtype).view(B, T, 1, 1)

        phi = self.phi.to(q.dtype).view(1, 1, 1, head_dim)
        gate = pos_gate[:, :T].to(q.dtype) * torch.cos(phi)

        q = q * gate * a
        k = k * gate * a

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

    def forward(self, x, pos_gate, token_is_mask):
        x = x + self.attn(norm(x), pos_gate, token_is_mask)
        x = x + self.mlp(norm(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # [NEW]: Polar (per-dim) positional gate for Q/K, cached as a buffer
        self.pos_seq_len = block_size * 2  # same spirit as old rotary_seq_len
        pos_gate = self._precompute_polar_gate(self.pos_seq_len)  # (1, L, 1, head_dim)
        self.register_buffer("pos_gate", pos_gate, persistent=False)

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

    def _precompute_polar_gate(self, seq_len, base=10000, device=None):
        """
        Polar per-dimension phase gate:
            gate[i, k] = cos(i * omega_k + phi_k)
        Here we cache cos(i * omega_k) and keep phi_k as learnable in the attention module.
        """
        if device is None:
            device = self.token_emb.weight.device

        k = torch.arange(head_dim, dtype=torch.float32, device=device)
        omega = 1.0 / (base ** (k / head_dim))

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        phase = torch.outer(t, omega)

        gate = torch.cos(phase)[None, :, None, :]
        return gate

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.size()

        x = self.token_emb(idx)
        x = norm(x)

        assert T <= self.pos_gate.size(1)
        pos_gate = self.pos_gate[:, :T]
        token_is_mask = idx == mask_token_id

        for block in self.blocks:
            x = block(x, pos_gate, token_is_mask)
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
    weights_path = os.getenv("TD_WEIGHTS_PATH", "weights/diffusion_polar.pt")
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
        start = time.time()
        output = generate(
            m, max_new_tokens=final_gen_tokens, temp=0.8, confidence_threshold=0.95, top_k=2
        )
        print(f"Total generation time: {time.time() - start:.2f} seconds")
        print(f"\nOutput:\n{output}")
