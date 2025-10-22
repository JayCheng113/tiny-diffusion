"""
Training script for character-level discrete diffusion model
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import DiffusionTransformer, DiffusionConfig


class DiscreteNoiseSchedule:
    """
    Simple noise schedule for discrete diffusion.
    At each timestep, we have a probability of replacing a token with a random token.
    """

    def __init__(self, num_timesteps, vocab_size):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size

        # Linear schedule: probability of corruption increases linearly
        self.corruption_probs = torch.linspace(0.0, 0.95, num_timesteps)

    def add_noise(self, x_0, t):
        """
        Add noise to clean tokens x_0 at timestep t
        Args:
            x_0: Clean tokens, shape (B, T)
            t: Timestep indices, shape (B,)
        Returns:
            x_t: Noisy tokens at timestep t
        """
        B, T = x_0.shape
        device = x_0.device

        # Get corruption probability for each sample (index on CPU, then move to device)
        corruption_prob = self.corruption_probs[t.cpu()].to(device)  # (B,)

        # Create mask: which tokens to corrupt
        mask = torch.rand(B, T, device=device) < corruption_prob.unsqueeze(1)  # (B, T)

        # Generate random tokens
        random_tokens = torch.randint(0, self.vocab_size, (B, T), device=device)

        # Replace masked positions with random tokens
        x_t = torch.where(mask, random_tokens, x_0)

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

    # Convert to tokens (simple ASCII encoding)
    tokens = torch.tensor([min(ord(c), 127) for c in text], dtype=torch.long)

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


def train_step(model, x_0, noise_schedule, optimizer):
    """
    Single training step
    Args:
        model: DiffusionTransformer model
        x_0: Clean tokens, shape (B, T)
        noise_schedule: Noise schedule object
        optimizer: Optimizer
    Returns:
        loss: Training loss
    """
    B, T = x_0.shape
    device = x_0.device

    # Sample random timesteps
    t = torch.randint(0, noise_schedule.num_timesteps, (B,), device=device)

    # Add noise to get x_t
    x_t = noise_schedule.add_noise(x_0, t)

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
    noise_schedule,
    optimizer,
    num_steps=10000,
    sample_interval=500,
    patience=500,
):
    """
    Main training loop with early stopping
    """
    model.train()

    best_loss = float("inf")
    steps_without_improvement = 0

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        # Get batch
        x_0 = next(data_loader)

        # Training step
        loss = train_step(model, x_0, noise_schedule, optimizer)

        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if steps_without_improvement >= patience:
            tqdm.write(
                f"\nEarly stopping at step {step + 1} (no improvement for {patience} steps)"
            )
            break

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss:.4f}", "best": f"{best_loss:.4f}"})

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
                # Decode samples
                text = "".join([chr(min(int(c), 127)) for c in samples[0]])
                tqdm.write(f"\n--- Sample at step {step + 1} ---")
                tqdm.write(text)
                tqdm.write("--- End sample ---\n")
            model.train()


def main():
    # Hyperparameters (matching the video)
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    n_embd = 384
    n_head = 6
    n_layer = 6
    num_steps = 32  # Number of diffusion steps
    # Note: dropout not used in this diffusion model architecture

    # Configuration
    config = DiffusionConfig(
        sequence_len=block_size,
        vocab_size=128,  # ASCII
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_timesteps=num_steps,  # Number of diffusion steps
    )

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = DiffusionTransformer(config).to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Noise schedule
    noise_schedule = DiscreteNoiseSchedule(
        num_timesteps=config.max_timesteps, vocab_size=config.vocab_size
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Data loader
    data_loader = get_data_loader(
        data_path="data.txt",  # You'll need to provide this
        batch_size=batch_size,
        seq_len=config.sequence_len,
        device=device,
    )

    # Train
    print("Starting training...\n")
    train(
        model=model,
        data_loader=data_loader,
        noise_schedule=noise_schedule,
        optimizer=optimizer,
        num_steps=max_iters,
        sample_interval=eval_interval,
        patience=max_iters,  # No early stopping
    )

    # Save model
    torch.save(model.state_dict(), "diffusion_model.pt")
    print("Model saved to diffusion_model.pt")


if __name__ == "__main__":
    main()
