"""
Training script for character-level discrete diffusion model
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import DiffusionTransformer, DiffusionConfig, encode_text, decode_tokens


class MaskedDiffusionSchedule:
    """
    Masked diffusion schedule for discrete diffusion.
    At each timestep, we have a probability of masking a token with [MASK].
    """

    def __init__(self, num_timesteps, mask_token_id):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id

        # Linear schedule: probability of masking increases linearly
        self.mask_probs = torch.linspace(0.0, 0.95, num_timesteps)

    def add_masks(self, x_0, t):
        """
        Add masks to tokens x_0 at timestep
        Args:
            x_0: Clean tokens, shape (B, T)
            t: Timestep indices, shape (B,)
        Returns:
            x_t: Masked tokens at timestep t
        """
        B, T = x_0.shape
        device = x_0.device

        # Get masking probability for each sample (index on CPU, then move to device)
        mask_prob = self.mask_probs[t.cpu()].to(device)  # (B,)

        # Create mask: which tokens to replace with [MASK]
        mask = torch.rand(B, T, device=device) < mask_prob.unsqueeze(1)  # (B, T)

        # Replace masked positions with mask token
        x_t = torch.where(mask, self.mask_token_id, x_0)

        return x_t


def get_data_loader(data_path, batch_size, seq_len, device):
    """
    Simple data loader for text data
    Args:
        data_path: Path to text file
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to load data on
    """
    # Read the text file
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Convert to tokens
    tokens = encode_text(text)

    # Create batches
    num_batches = len(tokens) // (batch_size * seq_len)
    tokens = tokens[: num_batches * batch_size * seq_len]
    tokens = tokens.view(batch_size, -1)

    # Generator function
    def data_generator():
        while True:
            for i in range(0, tokens.size(1) - seq_len, seq_len):
                batch = tokens[:, i : i + seq_len].to(device)
                yield batch

    return data_generator()


def train_step(model, x_0, mask_schedule, optimizer):
    """
    Single training step
    Args:
        model: DiffusionTransformer model
        x_0: Clean tokens, shape (B, T)
        mask_schedule: Mask schedule object
        optimizer: Optimizer
    Returns:
        loss: Training loss
    """
    B, _ = x_0.shape
    device = x_0.device

    # Sample random timesteps
    t = torch.randint(0, mask_schedule.num_timesteps, (B,), device=device)

    # Add mask to get x_t
    x_t = mask_schedule.add_mask(x_0, t)

    # Forward pass: predict the original tokens
    logits = model(x_t, t)  # (B, T, vocab_size)

    # Compute loss: cross-entropy between predicted and original tokens
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), x_0.view(-1), reduction="mean"
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    model,
    data_loader,
    mask_schedule,
    optimizer,
    num_steps=10000,
    sample_interval=500,
):
    """
    Main training loop
    """
    model.train()

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        # Get batch
        x_0 = next(data_loader)

        # Training step
        loss = train_step(model, x_0, mask_schedule, optimizer)

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        # Sample generation
        if (step + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                samples = model.sample(
                    batch_size=1,
                    seq_len=100,
                    num_steps=None,  # Use all timesteps
                    temperature=1.0,
                    device=model.get_device(),
                )
                # Decode samples to text
                text = decode_tokens(samples[0])
                tqdm.write(f"\n--- Sample at step {step + 1} ---")
                tqdm.write(text)
                tqdm.write("--- End sample ---\n")
            model.train()


def main():
    # Hyperparameters
    batch_size = 64
    max_iters = 10000
    eval_interval = 500
    learning_rate = 3e-4

    config = DiffusionConfig()  # default config

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model
    model = DiffusionTransformer(config).to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Masked diffusion schedule
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=config.diffusion_steps, mask_token_id=config.mask_token_id
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Data loader
    data_loader = get_data_loader(
        data_path="data/shakespeare.txt",
        batch_size=batch_size,
        seq_len=config.sequence_len,
        device=device,
    )

    # Train
    print("Starting training...\n")
    train(
        model=model,
        data_loader=data_loader,
        mask_schedule=mask_schedule,
        optimizer=optimizer,
        num_steps=max_iters,
        sample_interval=eval_interval,
    )

    # Save model
    import os

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/diffusion_model.pt")
    print("Model saved to weights/diffusion_model.pt")


if __name__ == "__main__":
    main()
