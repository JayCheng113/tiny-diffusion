"""
Visualize the generation process step by step
Shows diffusion denoising and/or GPT autoregressive generation
"""

import sys
import os
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

import diffusion
import diffusion_flow
import gpt

_ANIM_REF = None


def generate_diffusion_frames(
    model,
    num_blocks=5,
    prompt_len=16,
    temp=0.8,
    confidence_threshold=0.95,
    top_k=2,
):
    """
    Generate samples and capture each denoising step

    Args:
        model: The trained diffusion model
        num_blocks: Number of blocks to generate
        prompt_len: Length of initial prompt
        temp: Sampling temperature
        confidence_threshold: Confidence threshold for decoding
        top_k: Top-k sampling parameter

    Returns:
        List of (all_tokens, mask_for_all, block_idx) tuples for each frame
    """
    device = next(model.parameters()).device
    block_size = diffusion.block_size
    mask_token_id = diffusion.mask_token_id

    print(f"Pre-calculating {num_blocks} blocks for diffusion...")

    all_frames = []
    all_tokens_history = diffusion.data[:prompt_len].tolist()

    def capture_frame(all_tokens_history, x, masked, block_idx):
        """Build and capture full sequence frame"""
        full_tokens = torch.tensor(all_tokens_history, dtype=torch.long, device=device)
        full_tokens = torch.cat(
            [full_tokens, x[0, prompt_len : prompt_len + block_len]]
        )
        full_mask = torch.zeros(len(full_tokens), dtype=torch.bool, device=device)
        full_mask[len(all_tokens_history) :] = masked[
            0, prompt_len : prompt_len + block_len
        ]
        all_frames.append(
            (full_tokens.cpu().clone(), full_mask.cpu().clone(), block_idx)
        )

    for block_idx in range(num_blocks):
        # How many tokens to generate this block
        max_new_tokens = 240
        block_len = min(block_size - prompt_len, max_new_tokens)

        # Initialize: last prompt_len tokens + masks
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(
            all_tokens_history[-prompt_len:], device=device
        )

        # Track which positions need decoding
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        capture_frame(all_tokens_history, x, masked, block_idx)

        # Iteratively decode
        step = 0
        while masked.any():
            # Get predictions and confidences
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            # Decode high-confidence masked positions (or at least 1)
            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences, torch.tensor(-float("inf"), device=device)
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

            capture_frame(all_tokens_history, x, masked, block_idx)
            step += 1

        # Extract and append generated tokens for next block
        all_tokens_history.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    print(f"Diffusion: Generated {len(all_frames)} frames")
    return all_frames


def generate_flow_frames(
    model,
    num_blocks=5,
    prompt_len=16,
    temp=0.8,
    confidence_threshold=0.95,
    top_k=2,
    flow_step=0.6,
):
    """
    Generate samples and capture each decoding step for diffusion_flow model.
    """
    device = next(model.parameters()).device
    block_size = diffusion_flow.block_size
    mask_token_id = diffusion_flow.mask_token_id

    print(f"Pre-calculating {num_blocks} blocks for flow diffusion...")

    all_frames = []
    all_tokens_history = diffusion_flow.data[:prompt_len].tolist()

    def capture_frame(all_tokens_history, x, masked, block_idx):
        full_tokens = torch.tensor(all_tokens_history, dtype=torch.long, device=device)
        full_tokens = torch.cat(
            [full_tokens, x[0, prompt_len : prompt_len + block_len]]
        )
        full_mask = torch.zeros(len(full_tokens), dtype=torch.bool, device=device)
        full_mask[len(all_tokens_history) :] = masked[
            0, prompt_len : prompt_len + block_len
        ]
        all_frames.append(
            (full_tokens.cpu().clone(), full_mask.cpu().clone(), block_idx)
        )

    for block_idx in range(num_blocks):
        max_new_tokens = 240
        block_len = min(block_size - prompt_len, max_new_tokens)

        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(
            all_tokens_history[-prompt_len:], device=device
        )

        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        capture_frame(all_tokens_history, x, masked, block_idx)

        while masked.any():
            logits, _ = model(x)

            current_state = model.token_emb(x)
            remaining_ratio = masked.float().mean().item()
            t_value = 1.0 - remaining_ratio
            t_batch = torch.full((1,), t_value, device=device)
            flow_v = model.forward_flow(current_state, t_batch)
            refined_state = current_state + flow_step * flow_v

            flow_logits = refined_state @ model.token_emb.weight.t()
            flow_blend = min(0.8, 0.25 + 0.55 * remaining_ratio)
            blended_logits = (1.0 - flow_blend) * logits + flow_blend * flow_logits

            probs = F.softmax(blended_logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            adaptive_threshold = max(
                0.60, confidence_threshold - 0.25 * (1.0 - remaining_ratio)
            )
            decode_mask = (confidences >= adaptive_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences, torch.tensor(-float("inf"), device=device)
                )
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(
                1, block_size
            )
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask
            capture_frame(all_tokens_history, x, masked, block_idx)

        all_tokens_history.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    print(f"Flow diffusion: Generated {len(all_frames)} frames")
    return all_frames


def generate_gpt_output(model, max_new_tokens, prompt_len=16, temp=0.8):
    """Generate full GPT output once"""
    device = next(model.parameters()).device
    block_size = gpt.block_size

    print(f"Generating {max_new_tokens} tokens with GPT...")

    x = gpt.data[:prompt_len].unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        cur_context = x[:, -block_size:]
        logits, _ = model(cur_context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temp, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)

    return x[0].cpu()


def animate_diffusion(
    diffusion_frames, num_blocks, decode_fn=diffusion.decode, title_prefix="Diffusion", chars_per_row=64
):
    """Create animation for diffusion model only"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    text_obj = ax.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
        linespacing=1.2,
        multialignment="left",
    )

    title = fig.suptitle("", fontsize=12)

    def update(frame_idx):
        # Use the last frame if we're in the pause period
        actual_idx = min(frame_idx, len(diffusion_frames) - 1)
        frame_tokens, mask, block_idx = diffusion_frames[actual_idx]

        text_chars = []
        for idx in range(len(frame_tokens)):
            char = decode_fn([frame_tokens[idx].item()])
            if char == "\n":
                char = " "
            text_chars.append("█" if mask[idx] else char)

        continuous_text = "".join(text_chars)
        lines = [
            continuous_text[i : i + chars_per_row]
            for i in range(0, len(continuous_text), chars_per_row)
        ]
        text_obj.set_text("\n".join(lines))

        num_masked = mask.sum().item()
        title.set_text(
            f"{title_prefix} - Block {block_idx + 1}/{num_blocks} - Remaining: {num_masked} tokens"
        )

        return [text_obj, title]

    # Add pause frames at the end (50 frames = 500ms pause)
    pause_frames = 50
    total_frames = len(diffusion_frames) + pause_frames

    anim = FuncAnimation(
        fig, update, frames=total_frames, interval=10, blit=False, repeat=True
    )
    plt.tight_layout()
    return anim


def animate_comparison(
    diffusion_frames,
    gpt_tokens,
    num_blocks,
    prompt_len,
    diffusion_decode_fn=diffusion.decode,
    diffusion_title_prefix="Diffusion",
    chars_per_row=64,
):
    """Create animation comparing diffusion and GPT"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    text_obj1 = ax1.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
        linespacing=1.2,
        multialignment="left",
    )

    text_obj2 = ax2.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
        linespacing=1.2,
        multialignment="left",
    )

    def update(frame_idx):
        # Update diffusion (top)
        # Use the last frame if we're in the pause period
        diffusion_idx = min(frame_idx, len(diffusion_frames) - 1)
        frame_tokens, mask, block_idx = diffusion_frames[diffusion_idx]

        text_chars = []
        for idx in range(len(frame_tokens)):
            char = diffusion_decode_fn([frame_tokens[idx].item()])
            if char == "\n":
                char = " "
            text_chars.append("█" if mask[idx] else char)

        continuous_text = "".join(text_chars)
        lines = [
            continuous_text[i : i + chars_per_row]
            for i in range(0, len(continuous_text), chars_per_row)
        ]
        text_obj1.set_text("\n".join(lines))

        num_masked = mask.sum().item()
        ax1.set_title(
            f"{diffusion_title_prefix} - Block {block_idx + 1}/{num_blocks} - Remaining: {num_masked} tokens",
            fontsize=12,
            pad=-20,
            y=0.98,
        )

        # Update GPT (bottom) - show tokens one by one
        gpt_idx = min(frame_idx + prompt_len, len(gpt_tokens))
        visible_tokens = gpt_tokens[:gpt_idx]

        text_chars = [
            gpt.decode([t.item()]) if gpt.decode([t.item()]) != "\n" else " "
            for t in visible_tokens
        ]
        continuous_text = "".join(text_chars)
        lines = [
            continuous_text[i : i + chars_per_row]
            for i in range(0, len(continuous_text), chars_per_row)
        ]
        text_obj2.set_text("\n".join(lines))

        ax2.set_title(
            f"GPT - Token {gpt_idx}/{len(gpt_tokens)} (Autoregressive)",
            fontsize=12,
            pad=-20,
            y=0.98,
        )

        return [text_obj1, text_obj2]

    # Add pause frames at the end (50 frames = 500ms pause)
    pause_frames = 50
    max_frames = max(len(diffusion_frames), len(gpt_tokens) - prompt_len) + pause_frames

    anim = FuncAnimation(
        fig, update, frames=max_frames, interval=10, blit=False, repeat=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return anim


def main():
    parser = argparse.ArgumentParser(
        description="Visualize diffusion and/or GPT generation"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Show both diffusion and GPT animations"
    )
    parser.add_argument(
        "--flow",
        action="store_true",
        help="Visualize diffusion_flow model instead of diffusion baseline",
    )
    parser.add_argument(
        "--flow-step",
        type=float,
        default=0.6,
        help="Flow refinement step size for diffusion_flow visualization",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of blocks to generate (default: 5)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=16,
        help="Length of initial prompt (default: 16)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path to save animation (e.g. out.gif or out.mp4)",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    # Load and generate diffusion variant
    if args.flow:
        model_path = os.path.join(os.path.dirname(__file__), "weights", "diffusion_flow.pt")
        print(f"Loading flow diffusion model from {model_path}...")
        diffusion_model = diffusion_flow.Model().to(device)
        diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
        diffusion_model.eval()
        diffusion_frames = generate_flow_frames(
            diffusion_model, args.blocks, args.prompt_len, flow_step=args.flow_step
        )
        title_prefix = "Flow Diffusion"
    else:
        model_path = os.path.join(os.path.dirname(__file__), "weights", "diffusion.pt")
        print(f"Loading diffusion model from {model_path}...")
        diffusion_model = diffusion.Model().to(device)
        diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
        diffusion_model.eval()
        diffusion_frames = generate_diffusion_frames(
            diffusion_model, args.blocks, args.prompt_len
        )
        title_prefix = "Diffusion"

    if args.compare:
        # Calculate how many tokens diffusion generates
        current_block_size = (
            diffusion_flow.block_size if args.flow else diffusion.block_size
        )
        max_new_tokens = args.blocks * (current_block_size - args.prompt_len)

        # Load and generate GPT
        gpt_path = os.path.join(os.path.dirname(__file__), "weights", "gpt.pt")
        print(f"Loading GPT model from {gpt_path}...")
        gpt_model = gpt.Model().to(device)
        gpt_model.load_state_dict(torch.load(gpt_path, map_location=device))
        gpt_model.eval()

        gpt_tokens = generate_gpt_output(gpt_model, max_new_tokens, args.prompt_len)

        print("Done! Showing comparison animation...\n")
        decode_fn = diffusion_flow.decode if args.flow else diffusion.decode
        anim = animate_comparison(
            diffusion_frames,
            gpt_tokens,
            args.blocks,
            args.prompt_len,
            diffusion_decode_fn=decode_fn,
            diffusion_title_prefix=title_prefix,
        )
    else:
        print("Done! Showing diffusion animation...\n")
        decode_fn = diffusion_flow.decode if args.flow else diffusion.decode
        anim = animate_diffusion(
            diffusion_frames,
            args.blocks,
            decode_fn=decode_fn,
            title_prefix=title_prefix,
        )

    global _ANIM_REF
    _ANIM_REF = anim  # keep a strong reference to avoid premature GC warnings

    backend = matplotlib.get_backend().lower()
    is_headless = backend.endswith("agg") or not os.environ.get("DISPLAY")

    if args.save is not None:
        save_path = args.save
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        print(f"Saving animation to {save_path} ...")
        anim.save(save_path)
        print("Animation saved.")
        return

    if is_headless:
        warnings.warn(
            "No interactive display detected. Use --save out.gif to render animation to file."
        )
        return

    plt.show()


if __name__ == "__main__":
    main()
