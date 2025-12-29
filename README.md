# tiny-diffusion

A character-level language diffusion model for text generation trained on Tiny Shakespeare, in 365 lines of code! It is only 10.7 million parameters, so you can also try it out locally!

![Demo](animations/animation.gif)

This repo also contains a tiny gpt implementation in 313 lines of code. ~80% of the code between the two files are the exact same.

> This is `v2` of this project, which simplified the diffusion code from ~1,000 lines to ~400, and slightly altered the architecture. To view the original version, view the `old` branch.

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd tiny-diffusion

# Install dependencies (Python 3.10+)
uv sync
```

### Fetch Model Weights
To fetch the trained model weights for our `gpt` and `diffusion` implementations, run:
```bash
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
Both models use the first 16 characters of `data.txt` as the initial context. This a parameter for the `generate` function and can be changed.

### Training
To train both model from scratch, run:
```bash
uv run diffusion.py --train

uv run gpt.py --train
```
The `gpt` model trains for 5000 iterations while the `diffusion` model trains for 10000. The weights are saved to the `weights/` directory.

The reason why the diffusion model trains for twice as long is because half as many tokens count towards the loss during training (only masked tokens contribute to the loss).

### Visualization
Visualize the generation process step-by-step:

```bash
# Visualize diffusion model only
uv run animations/visualize.py

# Compare diffusion and GPT side-by-side
uv run animations/visualize.py --compare

# Generate more blocks
uv run animations/visualize.py --blocks 10
```


## How It Works

### GPT (Autoregressive)
- Predicts the next token given all previous tokens
- Uses **causal attention** (can only look at past tokens)
- Generates text **sequentially** (one token at a time, left-to-right)
- Training: minimize cross-entropy loss on next token prediction

### Diffusion (Non-Autoregressive)
- Predicts original tokens given partially masked sequences
- Uses **bidirectional attention** (can look at all tokens)
- Generates text **in parallel** (multiple tokens per step, based on confidence)
- Generates in blocks: fills in masked tokens iteratively, then moves to the next block
- Training: minimize cross-entropy loss on denoising masked tokens

### Key Modifications

The diffusion model makes **5 key changes** to the GPT architecture:

1. **Add mask token** to vocabulary (`_`) for representing unknown tokens
2. **Change attention** from causal to bidirectional (`is_causal=False`)
3. **Change generation** from sequential to confidence-based parallel decoding
4. **Change training objective** from next token prediction to unmasking
5. **Only masked tokens** contribute to the loss during training


## License

MIT
