import argparse
import importlib
import json
import os
from pathlib import Path

import torch
from inference_utils import (
    default_weights_for_module,
    run_generate_with_capture,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run generation with a saved preset.")
    parser.add_argument("--preset", required=True, help="Preset name in presets file")
    parser.add_argument(
        "--presets-file",
        default="configs/inference_presets.json",
        help="Path to presets JSON",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional checkpoint path override. Default: weights by module",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    presets_path = Path(args.presets_file)
    presets = json.loads(presets_path.read_text(encoding="utf-8"))
    if args.preset not in presets:
        raise KeyError(f"Preset not found: {args.preset}")

    config = presets[args.preset]
    module_name = config["module"]
    module = importlib.import_module(module_name)

    weights_path = args.weights or default_weights_for_module(module_name)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")

    set_seed(args.seed)
    model = module.Model().to(module.device)
    model.load_state_dict(torch.load(weights_path, map_location=module.device))
    model.eval()

    kwargs = {
        "max_new_tokens": int(config["max_new_tokens"]),
        "prompt_len": int(config["prompt_len"]),
        "temp": float(config["temp"]),
        "confidence_threshold": float(config["confidence_threshold"]),
        "top_k": int(config["top_k"]),
    }
    if "flow_step" in config:
        kwargs["flow_step"] = float(config["flow_step"])

    print(f"Preset: {args.preset}")
    print(f"Module: {module_name}")
    print(f"Weights: {weights_path}")
    print(f"Seed: {args.seed}")
    print(f"Params: {kwargs}")

    text, elapsed, total_steps, avg_decoded, _ = run_generate_with_capture(
        module, model, **kwargs
    )

    print(f"Elapsed: {elapsed:.4f}s")
    if total_steps >= 0:
        print(f"Total steps: {total_steps}")
    if avg_decoded >= 0:
        print(f"Avg decoded/step: {avg_decoded:.4f}")
    print("\nOutput:\n")
    print(text)


if __name__ == "__main__":
    main()
