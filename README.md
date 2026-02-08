# tiny-diffusion

A character-level language diffusion model for text generation trained on Tiny Shakespeare, in 365 lines of code! It is only 10.7 million parameters, so you can also try it out locally!

![Demo](https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/animation.gif)

This repo also contains a tiny gpt implementation in 313 lines of code. ~80% of the code between the two files are the exact same.

> This is `v2` of this project, which simplified the diffusion code from ~1,000 lines to ~400, and slightly altered the architecture. To view the original version, view the `old` branch.


## Quick Start

### Installation
```bash
# Install dependencies (Python 3.10+)
uv sync

# Download the dataset
wget https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/data.txt

# Download the trained model weights (if you don't want to train it from scratch)
mkdir -p weights && wget -P weights https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/{gpt,diffusion}.pt
```

### Generation
Generate text with the trained models:
```bash
# Diffusion (parallel decoding)
uv run diffusion.py

# GPT (autoregressive)
uv run gpt.py
```
Both models generate 2,000 characters by default and use the first 16 characters of `data.txt` as the initial context. These are parameters in the `generate` function and can be easily modified.

### Training
To train both models from scratch, run:

```bash
# Train diffusion model
uv run diffusion.py --train

# Train GPT model
uv run gpt.py --train
```

To train from a JSONL dataset, pass `--data` and the text field name:

```bash
# JSONL row example: {"text": "..."}
uv run diffusion.py --train --data your_dataset.jsonl --jsonl-field text
uv run gpt.py --train --data your_dataset.jsonl --jsonl-field text
```

If your JSONL uses another key (e.g. `content`), replace `--jsonl-field text` with that key.

The `gpt` model trains for 5,000 iterations while the `diffusion` model trains for 10,000, taking ~10 and ~20 minutes respectively on an A100 GPU. The weights are saved to the `weights/` directory.

The diffusion model trains for twice as long because half as many tokens count towards the loss during training (only masked tokens contribute to the loss).

### Visualization
Visualize the generation process step-by-step:

```bash
# Visualize diffusion model only
uv run visualize.py

# Compare diffusion and GPT side-by-side
uv run visualize.py --compare

# Generate more blocks
uv run visualize.py --blocks 10
```


## Differences Between The Models

### GPT (Autoregressive)
- Predicts the next token given all previous tokens
- Uses **causal attention** (can only look at past tokens)
- Generates text **sequentially** (one token at a time, left-to-right)
- Training: minimize cross-entropy loss on next token prediction

### Diffusion (Non-Autoregressive)
- Predicts original tokens given partially masked sequences
- Uses **bidirectional attention** (can look at all tokens)
- Generates text **in parallel** and in blocks: fills in masked tokens iteratively, then moves to the next block
- Training: minimize cross-entropy loss on denoising masked tokens

### Key Modifications
The diffusion model makes **5 key changes** to the GPT architecture:

1. **Add mask token** to vocabulary (`_`) for representing noised tokens
2. **Change attention** from causal to bidirectional (`is_causal=False`)
3. **Change generation** from sequential to confidence-based parallel decoding
4. **Change training objective** from next token prediction to unmasking
5. **Only masked tokens** contribute to the loss during training


## Acknoledgements
The code for `gpt.py` and `diffusion.py` take heavy inspiration from the Andrej Karpathy GPT implementations listed below:
- [nanochat GPT implementation](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)
- ["Let's build GPT" video GPT implementation](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)

My GPT implemenation, `gpt.py`, aims to strike a balance between simplicity and good generation quality.

The `diffusion.py` file is a modified version of `gpt.py` with as few modifications as possible to get it to do language diffusion.


## License

MIT
