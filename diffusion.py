# Code takes heavy inspiration from Andrej Karpathy's two implementations:
# nanochat: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
# "Let's build GPT" video: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import os
import time
import argparse
import json
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # default context length (can be overridden by --seq-len)
max_iters = 25000
eval_interval = 1000
learning_rate = 3e-4
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 8
head_dim = n_embd // n_head
ffn_intermediate_size = None
ffn_dropout = 0.0
ffn_hidden_act = "silu"
rope_base = 1e6
rms_norm_eps = 1e-5
max_position_embeddings = 32768
rope_scaling = {
    "original_max_position_embeddings": 2048,
    "factor": 16,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.0,
}
# ------------
torch.manual_seed(1337)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run tiny diffusion model")
    parser.add_argument("--train", action="store_true", help="Train from scratch")
    parser.add_argument("--hidden-size", type=int, default=n_embd)
    parser.add_argument("--num-hidden-layers", type=int, default=n_layer)
    parser.add_argument("--num-attention-heads", type=int, default=n_head)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=ffn_dropout)
    parser.add_argument("--hidden-act", default=ffn_hidden_act, choices=["relu", "gelu", "silu"])
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--rms-norm-eps", type=float, default=rms_norm_eps)
    parser.add_argument("--rope-theta", type=float, default=rope_base)
    parser.add_argument("--max-position-embeddings", type=int, default=max_position_embeddings)
    parser.add_argument("--inference-rope-scaling", action="store_true")
    parser.add_argument("--target-vocab-size", type=int, default=6400)
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        help="Use tokenizer.json/tokenizer_config.json via HuggingFace tokenizer",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=".",
        help="Directory containing tokenizer files",
    )
    parser.add_argument(
        "--data",
        default="data.txt",
        help="Text corpus path. Supports .txt and .jsonl",
    )
    parser.add_argument(
        "--jsonl-field",
        default="text",
        help="Field name to read from each jsonl row",
    )
    parser.add_argument(
        "--jsonl-sep",
        default="\n",
        help="Separator used when concatenating jsonl rows",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=block_size,
        help="Sequence length used for training/generation",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Optional checkpoint path for loading/saving weights",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=max_iters,
        help="Total training steps target (supports resume)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name used for checkpoint/loss filenames",
    )
    return parser.parse_args()


def load_text_corpus(path, jsonl_field, jsonl_sep):
    if path.endswith(".jsonl"):
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                value = row.get(jsonl_field, "")
                if isinstance(value, str) and value:
                    lines.append(value)
                elif value:
                    lines.append(str(value))
        if not lines:
            raise ValueError(
                f"No usable rows found in {path}. Check --jsonl-field {jsonl_field!r}."
            )
        return jsonl_sep.join(lines)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_text_samples(path, jsonl_field):
    if path.endswith(".jsonl"):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                value = row.get(jsonl_field, "")
                if isinstance(value, str) and value:
                    samples.append(value)
                elif value:
                    samples.append(str(value))
        if not samples:
            raise ValueError(
                f"No usable rows found in {path}. Check --jsonl-field {jsonl_field!r}."
            )
        return samples

    text = load_text_corpus(path, jsonl_field, "\n")
    if not text:
        raise ValueError(f"Empty text corpus in {path}.")
    return [text]


def find_unused_char(text):
    for code in [0, 1, 2, 3, 4, 5, 6, 7]:
        ch = chr(code)
        if ch not in text:
            return ch
    raise ValueError("Failed to find an unused mask token in corpus.")


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, text_field="text", max_length=512):
        super().__init__()
        self.data_path = data_path
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")
        if self.bos_token_id is None or self.eos_token_id is None:
            raise ValueError("Tokenizer must define bos_token_id and eos_token_id.")

        self.is_jsonl = data_path.endswith(".jsonl")
        if self.is_jsonl:
            self.offsets = []
            with open(data_path, "r", encoding="utf-8") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    value = row.get(text_field, "")
                    if isinstance(value, str) and value:
                        self.offsets.append(offset)
                    elif value:
                        self.offsets.append(offset)
            if not self.offsets:
                raise ValueError(
                    f"No usable rows found in {data_path}. Check --jsonl-field {text_field!r}."
                )
        else:
            self.samples = load_text_samples(data_path, text_field)

    def __len__(self):
        if self.is_jsonl:
            return len(self.offsets)
        return len(self.samples)

    def __getitem__(self, index):
        if self.is_jsonl:
            with open(self.data_path, "r", encoding="utf-8") as f:
                f.seek(self.offsets[index])
                row = json.loads(f.readline())
            value = row.get(self.text_field, "")
            sample = value if isinstance(value, str) else str(value)
        else:
            sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        input_ids = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        return torch.tensor(input_ids, dtype=torch.long)


def ensure_nonempty_mask(mask, candidate_mask):
    # Ensure at least one position is masked in each sample with valid tokens.
    for b in range(mask.size(0)):
        if candidate_mask[b].any() and not mask[b].any():
            valid_pos = torch.nonzero(candidate_mask[b], as_tuple=False).view(-1)
            chosen = valid_pos[torch.randint(valid_pos.numel(), (1,))]
            mask[b, chosen] = True
    return mask


args = parse_args()
block_size = args.seq_len
if block_size <= 0:
    raise ValueError("--seq-len must be > 0")
if args.hidden_size <= 0 or args.num_hidden_layers <= 0 or args.num_attention_heads <= 0:
    raise ValueError("hidden-size/num-hidden-layers/num-attention-heads must be > 0")
if args.hidden_size % args.num_attention_heads != 0:
    raise ValueError("hidden-size must be divisible by num-attention-heads")

n_embd = args.hidden_size
n_head = args.num_attention_heads
n_layer = args.num_hidden_layers
head_dim = n_embd // n_head
ffn_intermediate_size = args.intermediate_size
ffn_dropout = args.dropout
ffn_hidden_act = args.hidden_act
learning_rate = args.learning_rate
warmup_steps = args.warmup_steps
min_lr_ratio = args.min_lr_ratio
rms_norm_eps = args.rms_norm_eps
rope_base = args.rope_theta
max_position_embeddings = args.max_position_embeddings
max_iters = args.max_iters
if max_iters <= 0:
    raise ValueError("--max-iters must be > 0")
if learning_rate <= 0:
    raise ValueError("--learning-rate must be > 0")
if warmup_steps < 0:
    raise ValueError("--warmup-steps must be >= 0")
if not (0.0 <= min_lr_ratio <= 1.0):
    raise ValueError("--min-lr-ratio must be in [0, 1]")
rope_scaling = (
    {
        "original_max_position_embeddings": 2048,
        "factor": 16,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.0,
        "type": "yarn",
    }
    if args.inference_rope_scaling
    else None
)

if args.use_tokenizer:
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for --use-tokenizer. Install with: uv add transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    mask_token = "<|mask|>"
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    if args.target_vocab_size is not None and args.target_vocab_size > len(tokenizer):
        extra_count = args.target_vocab_size - len(tokenizer)
        tokenizer.add_tokens([f"<|extra_{i}|>" for i in range(extra_count)])
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    dataset = PretrainDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        text_field=args.jsonl_field,
        max_length=block_size,
    )
    if len(dataset) < 2:
        raise ValueError("Need at least 2 samples for train/val split in tokenizer mode.")
    train_size = int(0.9 * len(dataset))
    train_size = min(max(1, train_size), len(dataset) - 1)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1337),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    loader_iters = {"train": iter(train_loader), "val": iter(val_loader)}

    first_sample = train_dataset[0]
    prompt_tokens = first_sample[first_sample != pad_token_id].tolist()
    if len(prompt_tokens) < 2:
        raise ValueError("First training sample has too few non-pad tokens.")

    def decode(l):
        return tokenizer.decode(l, skip_special_tokens=False)

else:
    text = load_text_corpus(args.data, args.jsonl_field, args.jsonl_sep)
    if len(text) <= block_size:
        raise ValueError(
            f"Corpus too short ({len(text)} chars). Need > block_size={block_size}."
        )

    # All the unique characters that occur in this text
    chars = sorted(list(set(text)))
    mask_char = find_unused_char(text)
    chars = [mask_char] + chars
    vocab_size = len(chars)
    # Create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    mask_token_id = stoi[mask_char]
    pad_token_id = None

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
    prompt_tokens = data[:16].tolist()


# [NEW]: Modify get batch to do masking
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if args.use_tokenizer:
        loader = train_loader if split == "train" else val_loader
        try:
            x = next(loader_iters[split])
        except StopIteration:
            loader_iters[split] = iter(loader)
            x = next(loader_iters[split])
        x = x.clone()
        y = x.clone()  # original tokens
        candidate_mask = (
            (x != pad_token_id)
            & (x != tokenizer.bos_token_id)
            & (x != tokenizer.eos_token_id)
        )
    else:
        data = train_data if split == "train" else val_data
        idx = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in idx])
        y = x.clone()  # original tokens
        candidate_mask = torch.ones_like(x, dtype=torch.bool)

    # Mask tokens with random probability per sample
    bsz, seq_len = x.size()
    mask_probs = torch.rand(bsz, 1)
    mask = (torch.rand(bsz, seq_len) < mask_probs) & candidate_mask
    mask = ensure_nonempty_mask(mask, candidate_mask)
    x[mask] = mask_token_id

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),), eps=rms_norm_eps)


def get_activation(name):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


def get_lr(step):
    max_lr = learning_rate
    min_lr = learning_rate * min_lr_ratio

    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)

    if max_iters <= warmup_steps:
        return min_lr

    progress = float(step - warmup_steps) / float(max_iters - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def precompute_freqs_cis(dim, end, rope_base, rope_scaling=None, device=None):
    if device is None:
        device = "cpu"

    freqs = 1.0 / (
        rope_base ** (torch.arange(0, dim, 2, device=device).float()[: (dim // 2)] / dim)
    )
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-gamma) + gamma/s), gamma is a linear ramp.
            inv_dim = lambda b: (
                dim * math.log(orig_max / (b * 2 * math.pi)) / (2 * math.log(rope_base))
            )
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def rotate_half(x):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    out = (x * cos) + (rotate_half(x) * sin)
    return out.to(x.dtype)


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
        intermediate_size = ffn_intermediate_size
        if intermediate_size is None:
            intermediate_size = int(n_embd * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.dropout = nn.Dropout(ffn_dropout)
        self.act_fn = get_activation(ffn_hidden_act)

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


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
        self.rotary_seq_len = max(max_position_embeddings, block_size * 2)
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
        cos, sin = precompute_freqs_cis(
            dim=head_dim,
            end=seq_len,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            device=device,
        )
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
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
    model, max_new_tokens, prompt_len=16, temp=1.0, confidence_threshold=0.95, top_k=3
):
    effective_prompt_len = min(prompt_len, len(prompt_tokens))
    all_tokens = prompt_tokens[:effective_prompt_len]
    total_steps = 0

    # Generate one block at a time
    while len(all_tokens) - effective_prompt_len < max_new_tokens:
        # How many tokens to generate this block
        block_len = min(240, effective_prompt_len + max_new_tokens - len(all_tokens))

        # Initialize: last prompt_len tokens + masks
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :effective_prompt_len] = torch.tensor(
            all_tokens[-effective_prompt_len:], device=device
        )

        # Track which positions need decoding
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, effective_prompt_len : effective_prompt_len + block_len] = True

        # Iteratively decode
        while masked.any():
            total_steps += 1

            # Get predictions and confidences
            logits, _ = model(x)
            logits[..., mask_token_id] = -float("inf")
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
        all_tokens.extend(
            x[0, effective_prompt_len : effective_prompt_len + block_len].tolist()
        )

    tokens_generated = len(all_tokens) - effective_prompt_len
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
    train_flag = args.train
    if args.run_name:
        default_weights_path = os.path.join("weights", f"{args.run_name}.pt")
    else:
        default_weights_path = (
            "weights/diffusion_tokenizer.pt" if args.use_tokenizer else "weights/diffusion.pt"
        )
    weights_path = args.weights_path or default_weights_path
    weights_dir = os.path.dirname(weights_path)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)

    model = Model()
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step = 0
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_train_losses = []
    eval_val_losses = []
    checkpoint_loaded = False

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
    if args.use_tokenizer:
        print(
            f"Corpus: {args.data}, samples: {len(dataset)}, seq_len: {block_size}, vocab: {vocab_size}"
        )
    else:
        print(f"Corpus: {args.data}, chars: {len(text)}, vocab: {vocab_size}")
    print(
        f"LR schedule: warmup+cosine, max_lr={learning_rate:.2e}, "
        f"min_lr={learning_rate * min_lr_ratio:.2e}, warmup_steps={warmup_steps}"
    )

    if os.path.exists(weights_path):
        print(f"Loading checkpoint from {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            m.load_state_dict(ckpt["model_state_dict"])
            checkpoint_loaded = True
            if train_flag and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = int(ckpt.get("step", -1)) + 1 if train_flag else 0
            train_steps = ckpt.get("train_steps", train_steps)
            train_losses = ckpt.get("train_losses", train_losses)
            eval_steps = ckpt.get("eval_steps", eval_steps)
            eval_train_losses = ckpt.get("eval_train_losses", eval_train_losses)
            eval_val_losses = ckpt.get("eval_val_losses", eval_val_losses)
            if train_flag:
                print(f"Resuming training from step {start_step}")
        else:
            # Backward compatibility: old checkpoints saved as raw state_dict.
            m.load_state_dict(ckpt)
            checkpoint_loaded = True
            if train_flag:
                print("Loaded model weights (no optimizer/step state, resume from step 0)")
    elif not train_flag:
        raise FileNotFoundError(
            f"No checkpoint found at {weights_path}. Use --train to train from scratch."
        )

    if train_flag:
        if not checkpoint_loaded:
            print("Training from scratch")
        elif start_step >= max_iters:
            print(
                f"Checkpoint already at/after max_iters ({start_step} >= {max_iters}). Skipping training."
            )

        start = time.time()
        total_train_steps = max(max_iters - start_step, 0)
        pbar = (
            tqdm(total=total_train_steps, desc="Training", dynamic_ncols=True)
            if tqdm and total_train_steps > 0
            else None
        )
        last_step = start_step - 1
        for step in range(start_step, max_iters):
            last_step = step
            current_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # every once in a while evaluate the loss on train and val sets
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss()
                eval_steps.append(step)
                eval_train_losses.append(losses["train"].item())
                eval_val_losses.append(losses["val"].item())
                print(
                    f"step {step}: train loss {losses['train']:.4f},"
                    f"val loss {losses['val']:.4f}, lr {current_lr:.2e}, "
                    f"time {time.time() - start:.2f} seconds"
                )
                # Generate a sample
                sample = generate(m, max_new_tokens=240)
                print(f"Sample:\n{sample}\n")

            # sample a batch of data
            xb, yb, mb = get_batch("train")

            # evaluate the loss
            logits, loss = model(xb, yb, mb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_steps.append(step)
            train_losses.append(loss.item())

            if pbar is not None:
                pbar.update(1)
                if step % 10 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
            elif step % 100 == 0:
                elapsed = time.time() - start
                done = step - start_step + 1
                speed = done / max(elapsed, 1e-6)
                eta = (max_iters - step - 1) / max(speed, 1e-6)
                print(
                    f"progress {step + 1}/{max_iters}, "
                    f"loss {loss.item():.4f}, lr {current_lr:.2e}, eta {eta:.1f}s"
                )

        if pbar is not None:
            pbar.close()

        # Save the model weights
        print(f"Total training time: {time.time() - start:.2f} seconds")
        print(f"Saving checkpoint to {weights_path}")
        torch.save(
            {
                "model_state_dict": m.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": last_step,
                "train_steps": train_steps,
                "train_losses": train_losses,
                "eval_steps": eval_steps,
                "eval_train_losses": eval_train_losses,
                "eval_val_losses": eval_val_losses,
                "args": vars(args),
            },
            weights_path,
        )

        # Save loss curve
        if args.run_name:
            plot_path = os.path.join("weights", f"{args.run_name}_loss.png")
        else:
            plot_path = (
                "weights/diffusion_tokenizer_loss.png"
                if args.use_tokenizer
                else "weights/diffusion_loss.png"
            )
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_losses, label="train (per step)", alpha=0.35)
        plt.plot(eval_steps, eval_train_losses, label="train (eval avg)", linewidth=2)
        plt.plot(eval_steps, eval_val_losses, label="val (eval avg)", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved loss plot to {plot_path}")

    # generate from the model
    start = time.time()
    output = generate(
        m, max_new_tokens=2000, temp=0.8, confidence_threshold=0.95, top_k=2
    )
    print(f"Total generation time: {time.time() - start:.2f} seconds")
    print(f"\nOutput:\n{output}")
