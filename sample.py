"""
Sample/inference script for the trained diffusion model
"""

import torch
from model import DiffusionTransformer, DiffusionConfig, decode_tokens


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    # Create model with same config as training
    config = DiffusionConfig()

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

            # Decode tokens to text
            text = decode_tokens(tokens[0])

            print(f"--- Sample {i + 1} ---")
            print(text)
            print()


def generate_continuous_blocks(
    model, num_blocks=30, seq_len=256, num_steps=32, temperature=1.0, context_len=64
):
    """
    Generate multiple blocks sequentially, where each block is conditioned on the
    last context_len characters of the previous block.

    Args:
        model: The trained diffusion model
        num_blocks: Number of blocks to generate
        seq_len: Length of each sequence block
        num_steps: Number of denoising steps
        temperature: Sampling temperature
        context_len: Number of characters from previous block to condition on (default 32)
    """
    device = model.get_device()

    print(
        f"Generating {num_blocks} continuous blocks with {num_steps} denoising steps..."
    )
    print(
        f"Each block conditions on the last {context_len} characters of the previous block\n"
    )

    all_text = ""

    for block_idx in range(num_blocks):
        with torch.no_grad():
            if block_idx == 0:
                # First block: generate from scratch
                tokens = model.sample(
                    batch_size=1,
                    seq_len=seq_len,
                    num_steps=num_steps,
                    temperature=temperature,
                    device=device,
                )
            else:
                # Subsequent blocks: condition on last context_len tokens
                # Start from all mask tokens
                x = torch.full(
                    (1, seq_len),
                    model.config.mask_token_id,
                    dtype=torch.long,
                    device=device,
                )

                # Set the first context_len tokens to the last context_len tokens of previous block
                # These will never be masked during denoising
                x[0, :context_len] = prev_context

                # Create a mask that protects the first context_len tokens
                # Denoise step by step, but keep context fixed
                for t in reversed(range(num_steps)):
                    t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                    logits = model.forward(x, t_batch)

                    if t > 0:
                        # Sample from predicted distribution
                        probs = torch.nn.functional.softmax(
                            logits / temperature, dim=-1
                        )
                        x_new = torch.multinomial(
                            probs.view(-1, model.config.vocab_size), num_samples=1
                        ).view(1, seq_len)

                        # Keep the context fixed, only update the rest
                        x[:, context_len:] = x_new[:, context_len:]
                    else:
                        # Final step: take argmax
                        x_new = torch.argmax(logits, dim=-1)
                        x[:, context_len:] = x_new[:, context_len:]

                tokens = x

            # Store the last context_len tokens for next iteration
            prev_context = tokens[0, -context_len:]

            # Decode tokens to text
            text = decode_tokens(tokens[0])

            # Print in real-time
            if block_idx == 0:
                print(text, end="", flush=True)
                all_text += text
            else:
                # Don't repeat the context that was already printed
                new_text = text[context_len:]
                print(new_text, end="", flush=True)
                all_text += new_text

    print("\n")
    return all_text


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

    # Generate continuous blocks
    generate_continuous_blocks(
        model,
        num_blocks=30,
        seq_len=model.config.sequence_len,
        num_steps=model.config.diffusion_steps,
        temperature=1.0,
        context_len=64,
    )


if __name__ == "__main__":
    main()
