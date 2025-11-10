# tiny-diffusion

A character-level language diffusion model for text generation. The model is a modified version of the [nanochat gpt](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
) implemenation and is trained on Tiny Shakespeare.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tiny-diffusion

# Install dependencies (Python 3.10+)
uv sync
```

## Quick Start

We already have a trained model weights in `weights/diffusion_model.pt`. The sample and animations files load the model from this file.

### Train Your Own Model

```bash
# Train from scratch on Shakespeare
uv run training.py

# Training will save checkpoints to weights/diffusion_model.pt
```

### Generate Text

```bash
# Generate samples using the pre-trained model
uv run sample.py
```

### Visualize the Diffusion Process

```bash
# Watch the denoising process step-by-step
uv run animations/diffusion-process.py

# See Game of Life-inspired sampling
uv run animations/game-of-life.py
```

## Default Config

- **Parameters**: 10.7 million
- **Layers**: 6
- **Attention Heads**: 6
- **Embedding Dim**: 384
- **Sequence Length**: 256 characters
- **Diffusion Steps**: 128

## File Structure

```
tiny-diffusion/
├── model.py                    # Core diffusion transformer
├── training.py                 # Training script
├── sample.py                   # Text generation
├── data/
│   └── tiny_shakespeare.txt    # Training data
├── weights/
│   └── diffusion_model.pt      # Pre-trained weights
└── animations/
    ├── diffusion-process.py    # Denoising visualization
    └── game-of-life.py         # Game of Life sampling
```
