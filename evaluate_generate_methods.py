import argparse
import csv
import importlib
import inspect
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from inference_utils import run_generate_with_capture, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="A/B compare two generate() parameter sets on the same model weights."
    )
    parser.add_argument("--module", default="diffusion", help="Python module name")
    parser.add_argument("--weights", default="weights/diffusion.pt", help="Checkpoint path")
    parser.add_argument("--seeds", default="1337,2027,7,42,123")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=2)

    # Method A: approximate legacy single-threshold decode
    parser.add_argument("--a-name", default="old_like")
    parser.add_argument("--a-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--a-draft-threshold", type=float, default=0.85)
    parser.add_argument("--a-confirm-threshold", type=float, default=0.85)
    parser.add_argument("--a-replace-margin", type=float, default=1.0)
    parser.add_argument("--a-target-chunk-len", type=int, default=240)

    # Method B: two-stage draft+confirm decode
    parser.add_argument("--b-name", default="new_two_stage")
    parser.add_argument("--b-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--b-draft-threshold", type=float, default=0.70)
    parser.add_argument("--b-confirm-threshold", type=float, default=0.85)
    parser.add_argument("--b-replace-margin", type=float, default=0.0)
    parser.add_argument("--b-target-chunk-len", type=int, default=16)

    parser.add_argument("--output-dir", default="eval_generate_reports")
    return parser.parse_args()


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


def supports_kwargs(module, kwargs):
    sig = inspect.signature(module.generate)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return accepted


def build_method_kwargs(prefix: str, args):
    return {
        "confidence_threshold": getattr(args, f"{prefix}_confidence_threshold"),
        "draft_threshold": getattr(args, f"{prefix}_draft_threshold"),
        "confirm_threshold": getattr(args, f"{prefix}_confirm_threshold"),
        "replace_margin": getattr(args, f"{prefix}_replace_margin"),
        "target_chunk_len": getattr(args, f"{prefix}_target_chunk_len"),
    }


def run_method(module, model, method_name: str, seeds, common_kwargs, method_kwargs):
    rows = []
    call_kwargs = supports_kwargs(module, {**common_kwargs, **method_kwargs})
    for seed in seeds:
        set_seed(seed)
        text, elapsed, steps, avg_decoded, _ = run_generate_with_capture(
            module, model, **call_kwargs
        )
        rows.append(
            {
                "method": method_name,
                "seed": seed,
                "elapsed_sec": float(elapsed),
                "total_steps": int(steps),
                "avg_decoded_per_step": float(avg_decoded),
                "text_len": len(text),
                "text": text,
                "effective_kwargs": call_kwargs,
            }
        )
    return rows


def summarize(rows):
    def ms(values):
        return {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }

    return {
        "elapsed_sec": ms([r["elapsed_sec"] for r in rows]),
        "total_steps": ms([r["total_steps"] for r in rows]),
        "avg_decoded_per_step": ms([r["avg_decoded_per_step"] for r in rows]),
        "text_len": ms([r["text_len"] for r in rows]),
        "n": len(rows),
    }


def write_rows_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "seed",
                "elapsed_sec",
                "total_steps",
                "avg_decoded_per_step",
                "text_len",
                "text",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["method"],
                    r["seed"],
                    f"{r['elapsed_sec']:.6f}",
                    r["total_steps"],
                    f"{r['avg_decoded_per_step']:.6f}",
                    r["text_len"],
                    r["text"],
                ]
            )


def main():
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("No valid seeds provided")

    module = load_module(args.module)
    model = load_model(module, args.weights)

    common_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "prompt_len": args.prompt_len,
        "temp": args.temp,
        "top_k": args.top_k,
    }
    a_kwargs = build_method_kwargs("a", args)
    b_kwargs = build_method_kwargs("b", args)

    a_rows = run_method(module, model, args.a_name, seeds, common_kwargs, a_kwargs)
    b_rows = run_method(module, model, args.b_name, seeds, common_kwargs, b_kwargs)

    a_summary = summarize(a_rows)
    b_summary = summarize(b_rows)
    all_rows = a_rows + b_rows

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.module}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": vars(args),
        "module": args.module,
        "weights": str(Path(args.weights).resolve()),
        "method_summaries": {
            args.a_name: a_summary,
            args.b_name: b_summary,
        },
        "delta": {
            "elapsed_sec_mean_diff": b_summary["elapsed_sec"]["mean"]
            - a_summary["elapsed_sec"]["mean"],
            "total_steps_mean_diff": b_summary["total_steps"]["mean"]
            - a_summary["total_steps"]["mean"],
            "avg_decoded_per_step_mean_diff": b_summary["avg_decoded_per_step"]["mean"]
            - a_summary["avg_decoded_per_step"]["mean"],
        },
        "quick_read": {
            "faster_method": args.b_name
            if b_summary["elapsed_sec"]["mean"] < a_summary["elapsed_sec"]["mean"]
            else args.a_name,
            "fewer_steps_method": args.b_name
            if b_summary["total_steps"]["mean"] < a_summary["total_steps"]["mean"]
            else args.a_name,
            "higher_parallel_decode_method": args.b_name
            if b_summary["avg_decoded_per_step"]["mean"]
            > a_summary["avg_decoded_per_step"]["mean"]
            else args.a_name,
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "results.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_rows_csv(out_dir / "results.csv", all_rows)

    print(f"Saved report to: {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Rows: {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
