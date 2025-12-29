# tiny-diffusion

A character-level language diffusion model for text generation trained on Tiny Shakespeare. It has everything (model architecture, training, and generation) in `diffusion.py` in just 351 lines of code! It is only 10.7 million parameters, so you can also try it out locally!

It also contains a tiny gpt implementation, 'gpt.py', which is very similar to 'diffusion.py'. The model architecture has one line changed (`is_causal=False`), allowing the model to do bidirectional attention instead of causal attention. The `get_batch` and `generate` functions are also modified; the rest of the code (~80%) is exact same.

This is `v2` of this project. It simplified the diffusion code from 955 lines to 351. To view the old version, fetch the `old` branch.


## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tiny-diffusion

# Install dependencies (Python 3.10+)
uv sync
```

### Training

Train the GPT model:
```bash
uv run gpt.py --train
```

Train the diffusion model:
```bash
uv run diffusion.py --train
```

Both models train for 5000 iterations and save weights to the `weights/` directory.

### Generation

Generate text with the trained models:

```bash
# GPT (autoregressive)
uv run gpt.py

# Diffusion (parallel decoding)
uv run diffusion.py
```

Both models use the first 16 characters of `data.txt` as the initial context.

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

The diffusion model makes just **4 key changes** to the GPT architecture:

1. **Add mask token** to vocabulary (`_`) for representing unknown tokens
2. **Change attention** from causal to bidirectional (`is_causal=False`)
3. **Change generation** from sequential to confidence-based parallel decoding
4. **Change training objective** from next token prediction to unmasking

### Implementation Details

- **Architecture**: 6-layer transformer with 6 attention heads, 384 embedding dimensions
- **Context length**: 256 tokens
- **Training data**: Character-level tokenization on `data.txt`
- **Optimizer**: AdamW with learning rate 3e-4
- **Batch size**: 64 sequences


## License

MIT
