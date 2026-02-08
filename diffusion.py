# Code takes heavy inspiration from Andrej Karpathy's two implementations:
# nanochat: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
# "Let's build GPT" video: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import os
import time
import argparse
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # default context length (can be overridden by --seq-len)
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train or run tiny diffusion model")
    parser.add_argument("--train", action="store_true", help="Train from scratch")
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
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")
        if self.bos_token_id is None or self.eos_token_id is None:
            raise ValueError("Tokenizer must define bos_token_id and eos_token_id.")

        self.samples = load_text_samples(data_path, text_field)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
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

if args.use_tokenizer:
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for --use-tokenizer. Install with: uv add transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    mask_token = "<|mask|>"
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    pad_token_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    dataset = PretrainDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        text_field=args.jsonl_field,
        max_length=block_size,
    )
    all_data = torch.stack([dataset[i] for i in range(len(dataset))])
    if all_data.size(0) < 2:
        raise ValueError("Need at least 2 samples for train/val split in tokenizer mode.")
    n = int(0.9 * all_data.size(0))
    n = min(max(1, n), all_data.size(0) - 1)
    train_data = all_data[:n]
    val_data = all_data[n:]

    first_sample = train_data[0]
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
    data = train_data if split == "train" else val_data
    if args.use_tokenizer:
        idx = torch.randint(data.size(0), (batch_size,))
        x = data[idx].clone()
        y = x.clone()  # original tokens
        candidate_mask = (
            (x != pad_token_id)
            & (x != tokenizer.bos_token_id)
            & (x != tokenizer.eos_token_id)
        )
    else:
        idx = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in idx])
        y = x.clone()  # original tokens
        candidate_mask = torch.ones_like(x, dtype=torch.bool)

    # Mask tokens with random probability per sample
    mask_probs = torch.rand(batch_size, 1)
    mask = (torch.rand(batch_size, block_size) < mask_probs) & candidate_mask
    mask = ensure_nonempty_mask(mask, candidate_mask)
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
    weights_path = (
        "weights/diffusion_tokenizer.pt" if args.use_tokenizer else "weights/diffusion.pt"
    )
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    model = Model()
    m = model.to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
    if args.use_tokenizer:
        print(
            f"Corpus: {args.data}, samples: {all_data.size(0)}, seq_len: {block_size}, vocab: {vocab_size}"
        )
    else:
        print(f"Corpus: {args.data}, chars: {len(text)}, vocab: {vocab_size}")

    # Load weights if they exist and train flag not set
    if os.path.exists(weights_path) and not train_flag:
        print(f"Loading weights from {weights_path}")
        m.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Training from scratch")

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        start = time.time()
        for iter in range(max_iters):
            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                print(
                    f"step {iter}: train loss {losses['train']:.4f},"
                    f"val loss {losses['val']:.4f}, time {time.time() - start:.2f} seconds"
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

        # Save the model weights
        print(f"Total training time: {time.time() - start:.2f} seconds")
        print(f"Saving weights to {weights_path}")
        torch.save(m.state_dict(), weights_path)

    # generate from the model
    start = time.time()
    output = generate(
        m, max_new_tokens=2000, temp=0.8, confidence_threshold=0.95, top_k=2
    )
    print(f"Total generation time: {time.time() - start:.2f} seconds")
    print(f"\nOutput:\n{output}")
