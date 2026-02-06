import contextlib
import io
import random
import re
import time

import torch


def parse_list(raw, fn):
    return [fn(x.strip()) for x in raw.split(",") if x.strip()]


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


def run_generate_with_capture(module, model, **kwargs):
    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        text = module.generate(model, **kwargs)
    elapsed = time.perf_counter() - start
    raw_stdout = buf.getvalue()
    total_steps, avg_decoded = parse_generate_stdout(raw_stdout)
    return text, elapsed, total_steps, avg_decoded, raw_stdout


def default_weights_for_module(module_name: str):
    if module_name == "diffusion":
        return "weights/diffusion.pt"
    if module_name == "diffusion_flow":
        return "weights/diffusion_flow.pt"
    return f"weights/{module_name}.pt"
