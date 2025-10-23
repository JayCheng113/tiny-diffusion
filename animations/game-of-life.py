"""
Sample from diffusion model using Conway's Game of Life rules for masking
Displays 256 tokens as a 16x16 grid and applies Game of Life dynamics
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import DiffusionTransformer, DiffusionConfig, encode_text, decode_tokens


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    config = DiffusionConfig()  # default config

    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model


def apply_game_of_life_rules(grid):
    """
    Apply Conway's Game of Life rules to a binary grid
    Args:
        grid: Binary tensor of shape (H, W)
    Returns:
        next_grid: Next state according to Game of Life rules
    """
    H, W = grid.shape

    # Pad the grid to handle edges (circular for wrapping/toroidal topology)
    # Need to add batch dimension for circular padding to work
    padded = F.pad(grid.float().unsqueeze(0), (1, 1, 1, 1), mode="circular").squeeze(0)

    # Count neighbors for each cell
    neighbors = torch.zeros_like(grid, dtype=torch.float32)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            neighbors += padded[1 + i : H + 1 + i, 1 + j : W + 1 + j]

    # Apply Game of Life rules:
    # 1. Any live cell with 2-3 neighbors survives
    # 2. Any dead cell with exactly 3 neighbors becomes alive
    # 3. All other cells die or stay dead
    next_grid = torch.zeros_like(grid)
    next_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
    next_grid[(grid == 0) & (neighbors == 3)] = 1

    return next_grid


def tokens_to_grid(tokens):
    """Convert 256 tokens to 16x16 binary grid based on character value"""
    # Use odd/even or threshold to create binary grid
    grid = (tokens % 2).reshape(16, 16)
    return grid


def display_grid(grid, tokens, mask=None):
    """Display the 16x16 grid with characters"""
    print("=" * 18)
    for i in range(16):
        row = ""
        for j in range(16):
            idx = i * 16 + j
            # Show ⬛ for masked positions (alive cells)
            if mask is not None and mask[idx]:
                row += "⬛"
            else:
                # Decode single token
                char = decode_tokens([tokens[idx]])
                # Replace newline with space
                if char == "\n":
                    char = " "
                row += char
        print("|" + row + "|")
    print("=" * 18)


def load_initial_text(data_path, num_chars=1024):
    """Load initial text from data file"""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()[:num_chars]

    # Pad if necessary
    if len(text) < num_chars:
        text = text + " " * (num_chars - len(text))

    # Convert to tokens
    tokens = encode_text(text)
    return tokens


def generate_with_game_of_life(
    model, initial_tokens, num_iterations=10, seq_len=256, num_steps=32, temperature=1.0
):
    """
    Generate samples using Game of Life rules to determine masking
    Uses matplotlib animation for live visualization
    Works on 32x32 grid (1024 tokens) by processing in 256-token chunks

    Args:
        model: The trained diffusion model
        initial_tokens: Initial 1024 tokens (32x32 grid)
        num_iterations: Number of Game of Life iterations to run
        seq_len: Sequence length for model (256)
        num_steps: Number of diffusion denoising steps
        temperature: Sampling temperature
    """
    assert len(initial_tokens) == 1024, "initial_tokens must be 1024 for 32x32 grid"

    device = model.get_device()
    tokens = initial_tokens.to(device)

    print(f"Pre-calculating {num_iterations} iterations with Game of Life dynamics...")

    # Convert to 32x32 grid with randomization for different initial patterns each time
    # Add masks to tokens before computing binary grid
    random_offset = torch.randint(0, 256, (1024,), device=device)
    grid = ((tokens + random_offset) % 2).reshape(32, 32)

    # Pre-calculate all frames
    all_frames = []
    all_masks = []

    # Store initial state
    all_frames.append(tokens.clone())
    all_masks.append((grid == 1).flatten())

    # Calculate all iterations
    for iteration in range(num_iterations):
        print(f"Calculating iteration {iteration + 1}/{num_iterations}...")

        # Apply Game of Life rules to get next state (on 32x32 grid)
        next_grid = apply_game_of_life_rules(grid)

        # Create mask: alive cells will be resampled
        mask = (next_grid == 1).flatten()  # Shape: (1024,)

        # Process all 1024 tokens in chunks of 256
        updated_tokens = tokens.clone()

        # Split into 4 chunks of 256 tokens each
        num_chunks = 1024 // seq_len
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * seq_len
            end_idx = start_idx + seq_len

            # Get chunk mask
            chunk_mask = mask[start_idx:end_idx]

            # Only process if there are masked positions in this chunk
            if chunk_mask.any():
                # Get chunk tokens and mask alive cells
                x = (
                    updated_tokens[start_idx:end_idx].clone().unsqueeze(0)
                )  # Shape: (1, 256)
                # Replace alive cells with mask token
                x[0, chunk_mask] = model.config.mask_token_id

                # Denoise the masked positions (alive cells)
                with torch.no_grad():
                    for t in reversed(range(num_steps)):
                        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                        logits = model.forward(x, t_batch)

                        if t > 0:
                            # Sample from predicted distribution
                            probs = F.softmax(logits / temperature, dim=-1)
                            x_new = torch.multinomial(
                                probs.view(-1, model.config.vocab_size), num_samples=1
                            ).view(1, seq_len)

                            # Only update masked positions (alive cells)
                            x[0, chunk_mask] = x_new[0, chunk_mask]
                        else:
                            # Final step: take argmax
                            x_new = torch.argmax(logits, dim=-1)
                            x[0, chunk_mask] = x_new[0, chunk_mask]

                # Update the chunk in the full token array
                updated_tokens[start_idx:end_idx] = x[0]

        # Update state
        tokens = updated_tokens
        grid = next_grid

        # Store frame
        all_frames.append(tokens.clone().cpu())
        all_masks.append(mask.cpu())

    print("Done! Now showing animation...\n")

    # Setup matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Create a SINGLE text object for all 32 rows (centered)
    # This ensures zero gap between lines
    text_obj = ax.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        fontweight="normal",
        linespacing=1.0,
        stretch="expanded",
    )

    title = fig.suptitle("Initial state - Masked cells: 0/1024", fontsize=14)

    def init():
        """Initialize animation"""
        # Show initial state
        frame_tokens = all_frames[0]
        mask = all_masks[0]

        # Build all rows as a single multi-line string (32x32 grid)
        lines = []
        for row_idx in range(32):
            row_text = ""
            for col_idx in range(32):
                idx = row_idx * 32 + col_idx
                if mask[idx]:
                    row_text += "■"
                else:
                    # Decode single token
                    char = decode_tokens([frame_tokens[idx]])
                    row_text += char
            lines.append(row_text)

        text_obj.set_text("\n".join(lines))
        text_obj.set_color("black")

        title.set_text(f"Initial state - Masked cells: {mask.sum().item()}/1024")
        return [text_obj, title]

    def update(frame_idx):
        """Update function for animation - just display pre-calculated frame"""
        if frame_idx >= len(all_frames):
            return [text_obj, title]

        frame_tokens = all_frames[frame_idx]
        mask = all_masks[frame_idx]

        # Update display - build all rows as a single multi-line string (32x32 grid)
        lines = []
        for row_idx in range(32):
            row_text = ""
            for col_idx in range(32):
                idx = row_idx * 32 + col_idx
                if mask[idx]:
                    row_text += "■"
                else:
                    # Decode single token
                    char = decode_tokens([frame_tokens[idx]])
                    row_text += char
            lines.append(row_text)

        text_obj.set_text("\n".join(lines))
        text_obj.set_color("black")

        title.set_text(
            f"Iteration {frame_idx} - Masked cells: {mask.sum().item()}/1024"
        )

        return [text_obj, title]

    # Create animation (faster, with looping)
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(all_frames),
        interval=50,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()

    return all_frames[-1]


def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # Load model
    checkpoint_path = "weights/diffusion_model.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print("Model loaded!\n")

    # Load initial text (first 1024 characters from data/shakespeare.txt)
    print("Loading initial text from data/shakespeare.txt...")
    initial_tokens = load_initial_text("data/shakespeare.txt", num_chars=1024)
    print(f"Loaded {len(initial_tokens)} characters\n")

    # Generate with Game of Life dynamics
    generate_with_game_of_life(
        model,
        initial_tokens,
        num_iterations=100,
        seq_len=model.config.sequence_len,
        num_steps=model.config.diffusion_steps,
        temperature=1.0,
    )


if __name__ == "__main__":
    main()
