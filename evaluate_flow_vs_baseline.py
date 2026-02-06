import argparse
import contextlib
import csv
import importlib
import io
import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class SampleMetrics:
    seed: int
    model_name: str
    elapsed_sec: float
    total_steps: int
    avg_decoded_per_step: float
    text_len: int
    underscore_ratio: float
    distinct_2: float
    repeat_char_ratio: float
    text: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare diffusion baseline vs flow variant."
    )
    parser.add_argument("--baseline-module", default="diffusion")
    parser.add_argument("--flow-module", default="diffusion_flow")
    parser.add_argument("--baseline-weights", default="weights/diffusion.pt")
    parser.add_argument("--flow-weights", default="weights/diffusion_flow.pt")
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--seeds", default="1337,2027,7")
    parser.add_argument("--output-dir", default="eval_reports")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(module_name: str):
    return importlib.import_module(module_name)


def load_model(module, weights_path: str):
    model = module.Model().to(module.device)
    ckpt_path = Path(weights_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    state = torch.load(str(ckpt_path), map_location=module.device)
    model.load_state_dict(state)
    model.eval()
    return model


def eval_val_loss(module, model, eval_batches: int, eval_batch_size: int):
    prev_bs = module.batch_size
    module.batch_size = eval_batch_size
    losses = []
    try:
        with torch.no_grad():
            for _ in range(eval_batches):
                x, y, m = module.get_batch("val")
                _, loss = model(x, y, m)
                losses.append(float(loss.item()))
    finally:
        module.batch_size = prev_bs

    return {
        "mean": float(statistics.mean(losses)),
        "std": float(statistics.pstdev(losses)) if len(losses) > 1 else 0.0,
        "n": len(losses),
    }


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


def text_metrics(text: str):
    n = len(text)
    if n == 0:
        return {
            "text_len": 0,
            "underscore_ratio": 0.0,
            "distinct_2": 0.0,
            "repeat_char_ratio": 0.0,
        }

    underscore_ratio = text.count("_") / n
    if n >= 2:
        bigrams = [text[i : i + 2] for i in range(n - 1)]
        distinct_2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct_2 = 0.0

    repeats = sum(1 for i in range(1, n) if text[i] == text[i - 1])
    repeat_char_ratio = repeats / max(n - 1, 1)
    return {
        "text_len": n,
        "underscore_ratio": float(underscore_ratio),
        "distinct_2": float(distinct_2),
        "repeat_char_ratio": float(repeat_char_ratio),
    }


def run_generate(module, model, seed: int, args):
    set_seed(seed)
    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        text = module.generate(
            model,
            max_new_tokens=args.max_new_tokens,
            prompt_len=args.prompt_len,
            temp=args.temp,
            confidence_threshold=args.confidence_threshold,
            top_k=args.top_k,
        )
    elapsed = time.perf_counter() - start
    raw_stdout = buf.getvalue()
    steps, avg_decoded = parse_generate_stdout(raw_stdout)
    tm = text_metrics(text)
    return SampleMetrics(
        seed=seed,
        model_name=module.__name__,
        elapsed_sec=float(elapsed),
        total_steps=steps,
        avg_decoded_per_step=avg_decoded,
        text_len=tm["text_len"],
        underscore_ratio=tm["underscore_ratio"],
        distinct_2=tm["distinct_2"],
        repeat_char_ratio=tm["repeat_char_ratio"],
        text=text,
    )


def aggregate_samples(samples):
    def ms(values):
        return {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }

    return {
        "elapsed_sec": ms([s.elapsed_sec for s in samples]),
        "total_steps": ms([s.total_steps for s in samples]),
        "avg_decoded_per_step": ms([s.avg_decoded_per_step for s in samples]),
        "text_len": ms([s.text_len for s in samples]),
        "underscore_ratio": ms([s.underscore_ratio for s in samples]),
        "distinct_2": ms([s.distinct_2 for s in samples]),
        "repeat_char_ratio": ms([s.repeat_char_ratio for s in samples]),
        "n": len(samples),
    }


def write_samples_csv(path: Path, samples):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "seed",
                "elapsed_sec",
                "total_steps",
                "avg_decoded_per_step",
                "text_len",
                "underscore_ratio",
                "distinct_2",
                "repeat_char_ratio",
                "text",
            ]
        )
        for s in samples:
            w.writerow(
                [
                    s.model_name,
                    s.seed,
                    f"{s.elapsed_sec:.6f}",
                    s.total_steps,
                    f"{s.avg_decoded_per_step:.6f}",
                    s.text_len,
                    f"{s.underscore_ratio:.6f}",
                    f"{s.distinct_2:.6f}",
                    f"{s.repeat_char_ratio:.6f}",
                    s.text,
                ]
            )


def write_blind_review(output_dir: Path, baseline_samples, flow_samples):
    rng = random.Random(20260206)
    pairs = list(zip(baseline_samples, flow_samples))
    blind_path = output_dir / "blind_samples.csv"
    key_path = output_dir / "blind_key.csv"

    with blind_path.open("w", newline="", encoding="utf-8") as bf, key_path.open(
        "w", newline="", encoding="utf-8"
    ) as kf:
        bw = csv.writer(bf)
        kw = csv.writer(kf)
        bw.writerow(["id", "seed", "sample_a", "sample_b"])
        kw.writerow(["id", "sample_a_model", "sample_b_model"])
        for i, (b, f) in enumerate(pairs):
            if rng.random() < 0.5:
                a_text, b_text = b.text, f.text
                a_model, b_model = b.model_name, f.model_name
            else:
                a_text, b_text = f.text, b.text
                a_model, b_model = f.model_name, b.model_name
            sid = f"pair_{i:03d}"
            bw.writerow([sid, b.seed, a_text, b_text])
            kw.writerow([sid, a_model, b_model])


def main():
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_module = load_module(args.baseline_module)
    flow_module = load_module(args.flow_module)

    baseline_model = load_model(baseline_module, args.baseline_weights)
    flow_model = load_model(flow_module, args.flow_weights)

    baseline_val = eval_val_loss(
        baseline_module, baseline_model, args.eval_batches, args.eval_batch_size
    )
    flow_val = eval_val_loss(flow_module, flow_model, args.eval_batches, args.eval_batch_size)

    baseline_samples = [run_generate(baseline_module, baseline_model, s, args) for s in seeds]
    flow_samples = [run_generate(flow_module, flow_model, s, args) for s in seeds]

    baseline_agg = aggregate_samples(baseline_samples)
    flow_agg = aggregate_samples(flow_samples)

    summary = {
        "config": vars(args),
        "baseline_module": args.baseline_module,
        "flow_module": args.flow_module,
        "val_loss": {
            args.baseline_module: baseline_val,
            args.flow_module: flow_val,
        },
        "generation": {
            args.baseline_module: baseline_agg,
            args.flow_module: flow_agg,
        },
        "quick_read": {
            "better_val_loss": args.flow_module
            if flow_val["mean"] < baseline_val["mean"]
            else args.baseline_module,
            "faster_generation": args.flow_module
            if flow_agg["elapsed_sec"]["mean"] < baseline_agg["elapsed_sec"]["mean"]
            else args.baseline_module,
            "fewer_steps": args.flow_module
            if flow_agg["total_steps"]["mean"] < baseline_agg["total_steps"]["mean"]
            else args.baseline_module,
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_samples_csv(out_dir / "samples_baseline.csv", baseline_samples)
    write_samples_csv(out_dir / "samples_flow.csv", flow_samples)
    write_blind_review(out_dir, baseline_samples, flow_samples)

    print(f"Saved report to: {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Blind samples: {out_dir / 'blind_samples.csv'}")
    print(f"Blind key: {out_dir / 'blind_key.csv'}")


if __name__ == "__main__":
    main()
