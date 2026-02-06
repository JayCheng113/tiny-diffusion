import argparse
import importlib
import json
import os
import random
import re
import time
from pathlib import Path

import torch


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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_generate_stdout(raw: str):
    total_steps = -1
    avg_decoded = -1.0
    m_steps = re.search(r"Total steps:\s*(\d+)", raw)
    if m_steps:
        total_steps = int(m_steps.group(1))
    m_avg = re.search(r"Avg decoded per step:\s*([0-9]*\.?[0-9]+)", raw)
    if m_avg:
        avg_decoded = float(m_avg.group(1))
    return total_steps, avg_decoded


def default_weights_for_module(module_name: str):
    if module_name == "diffusion":
        return "weights/diffusion.pt"
    if module_name == "diffusion_flow":
        return "weights/diffusion_flow.pt"
    return f"weights/{module_name}.pt"


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

    import io
    import contextlib

    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        text = module.generate(model, **kwargs)
    elapsed = time.perf_counter() - start
    gen_stdout = buf.getvalue()
    total_steps, avg_decoded = parse_generate_stdout(gen_stdout)

    print(f"Elapsed: {elapsed:.4f}s")
    if total_steps >= 0:
        print(f"Total steps: {total_steps}")
    if avg_decoded >= 0:
        print(f"Avg decoded/step: {avg_decoded:.4f}")
    print("\nOutput:\n")
    print(text)


if __name__ == "__main__":
    main()
