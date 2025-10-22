"""
Sample/inference script for the trained diffusion model
"""

import torch
from model import DiffusionTransformer, DiffusionConfig


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    # Create model with same config as training
    config = DiffusionConfig(
        sequence_len=256,
        vocab_size=128,
        n_layer=6,
        n_head=8,
        n_embd=256,
        max_timesteps=16,
    )

    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model


def generate_samples(model, num_samples=5, seq_len=256, num_steps=16, temperature=1.0):
    """Generate text samples from the model"""
    device = model.get_device()

    print(f"Generating {num_samples} samples with {num_steps} denoising steps...\n")

    for i in range(num_samples):
        with torch.no_grad():
            # Generate a sample
            tokens = model.sample(
                batch_size=1,
                seq_len=seq_len,
                num_steps=num_steps,
                temperature=temperature,
                device=device,
            )

            # Decode to text
            text = "".join([chr(min(int(c), 127)) for c in tokens[0]])

            print(f"--- Sample {i + 1} ---")
            print(text)
            print()


def main():
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model
    checkpoint_path = "diffusion_model.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print("Model loaded!\n")

    # Generate samples
    generate_samples(
        model,
        num_samples=5,
        seq_len=256,
        num_steps=16,  # Use all diffusion steps
        temperature=1.0,  # Higher = more random, lower = more deterministic
    )


if __name__ == "__main__":
    main()
