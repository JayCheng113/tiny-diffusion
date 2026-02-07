import argparse
import hashlib
import importlib
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from inference_utils import parse_list, run_generate_with_capture, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep new_two_stage generate params and compare against baseline."
    )
    p.add_argument("--module", default="diffusion")
    p.add_argument("--weights", default="weights/diffusion.pt")
    p.add_argument("--seeds", default="1337,2027,7,42,123,314,2718,9001,65537,8888")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=2)

    # Baseline (old_like) single config
    p.add_argument("--base-confidence-threshold", type=float, default=0.85)
    p.add_argument("--base-draft-threshold", type=float, default=0.85)
    p.add_argument("--base-confirm-threshold", type=float, default=0.85)
    p.add_argument("--base-replace-margin", type=float, default=1.0)
    p.add_argument("--base-target-chunk-len", type=int, default=240)

    # Sweep (new_two_stage) grid
    p.add_argument("--sweep-confidence-threshold", default="0.95")
    p.add_argument("--sweep-draft-threshold", default="0.20,0.30,0.40,0.50,0.60,0.70")
    p.add_argument("--sweep-confirm-threshold", default="0.85,0.88")
    p.add_argument("--sweep-replace-margin", default="0.0,0.02,0.05")
    p.add_argument("--sweep-target-chunk-len", default="240")

    p.add_argument("--output-dir", default="sweep_generate_reports")
    return p.parse_args()


def load_module(module_name: str):
    return importlib.import_module(module_name)


def load_model(module, weights_path: str):
    model = module.Model().to(module.device)
    ckpt = Path(weights_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    state = torch.load(str(ckpt), map_location=module.device)
    model.load_state_dict(state)
    model.eval()
    return model


def text_metrics(text: str):
    n = len(text)
    if n == 0:
        return {
            "text_len": 0,
            "underscore_ratio": 0.0,
            "distinct_2": 0.0,
            "repeat_char_ratio": 0.0,
            "char_entropy": 0.0,
        }
    underscore_ratio = text.count("_") / n
    grams2 = [text[i : i + 2] for i in range(n - 1)] if n >= 2 else []
    distinct_2 = len(set(grams2)) / len(grams2) if grams2 else 0.0
    repeats = sum(1 for i in range(1, n) if text[i] == text[i - 1]) / max(n - 1, 1)
    cnt = Counter(text)
    entropy = 0.0
    for c in cnt.values():
        p = c / n
        entropy -= p * math.log(p, 2)
    return {
        "text_len": n,
        "underscore_ratio": float(underscore_ratio),
        "distinct_2": float(distinct_2),
        "repeat_char_ratio": float(repeats),
        "char_entropy": float(entropy),
    }


def run_one(module, model, seed: int, kwargs):
    set_seed(seed)
    text, elapsed, steps, avg_decoded, _ = run_generate_with_capture(module, model, **kwargs)
    tm = text_metrics(text)
    return {
        "seed": seed,
        "elapsed_sec": float(elapsed),
        "total_steps": int(steps),
        "avg_decoded_per_step": float(avg_decoded),
        **tm,
    }


def mean(values):
    return float(statistics.mean(values))


def config_id(cfg):
    key = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def print_progress(done: int, total: int, prefix: str = "Progress"):
    if total <= 0:
        return
    width = 30
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r{prefix} [{bar}] {done}/{total} ({ratio * 100:5.1f}%)"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


def draw_speed_vs_quality(agg_rows, out_path: Path):
    xs = [r["delta_elapsed_sec_mean"] for r in agg_rows]
    ys = [r["delta_distinct_2_mean"] for r in agg_rows]
    cs = [r["delta_repeat_char_ratio_mean"] for r in agg_rows]
    labels = [r["config_id"] for r in agg_rows]
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(xs, ys, c=cs, cmap="coolwarm", s=80, alpha=0.9, edgecolors="k")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    for i, lab in enumerate(labels[:8]):
        plt.annotate(lab, (xs[i], ys[i]), fontsize=8, xytext=(5, 3), textcoords="offset points")
    plt.xlabel("Delta Elapsed Seconds (new - baseline, lower is better)")
    plt.ylabel("Delta Distinct-2 (new - baseline, higher is better)")
    plt.title("Speed vs Quality (color: Delta Repeat-Char Ratio)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Delta Repeat-Char Ratio (lower is better)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def draw_speed_vs_parallel(agg_rows, out_path: Path):
    xs = [r["delta_elapsed_sec_mean"] for r in agg_rows]
    ys = [r["delta_avg_decoded_per_step_mean"] for r in agg_rows]
    labels = [r["config_id"] for r in agg_rows]
    plt.figure(figsize=(9, 6))
    plt.scatter(xs, ys, s=90, alpha=0.9, edgecolors="k")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    for i, lab in enumerate(labels[:8]):
        plt.annotate(lab, (xs[i], ys[i]), fontsize=8, xytext=(5, 3), textcoords="offset points")
    plt.xlabel("Delta Elapsed Seconds (new - baseline, lower is better)")
    plt.ylabel("Delta Avg Decoded per Step (higher is better)")
    plt.title("Speed vs Parallel Decode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def draw_topk_speed_bars(agg_rows, out_path: Path, top_k: int = 8):
    top = agg_rows[: min(top_k, len(agg_rows))]
    labels = [
        f"{r['config_id']}\n(d={r['draft_threshold']},c={r['confirm_threshold']},m={r['replace_margin']})"
        for r in top
    ]
    values = [r["delta_elapsed_sec_mean"] for r in top]
    plt.figure(figsize=(11, 6))
    plt.bar(range(len(top)), values)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(range(len(top)), labels, rotation=25, ha="right")
    plt.ylabel("Delta Elapsed Seconds (new - baseline)")
    plt.title("Top Configs by Speed Gain")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    seeds = parse_list(args.seeds, int)
    if not seeds:
        raise ValueError("No valid seeds provided")

    module = load_module(args.module)
    model = load_model(module, args.weights)

    common = {
        "max_new_tokens": args.max_new_tokens,
        "prompt_len": args.prompt_len,
        "temp": args.temp,
        "top_k": args.top_k,
    }
    baseline_cfg = {
        "confidence_threshold": args.base_confidence_threshold,
        "draft_threshold": args.base_draft_threshold,
        "confirm_threshold": args.base_confirm_threshold,
        "replace_margin": args.base_replace_margin,
        "target_chunk_len": args.base_target_chunk_len,
    }

    sweep_conf = parse_list(args.sweep_confidence_threshold, float)
    sweep_draft = parse_list(args.sweep_draft_threshold, float)
    sweep_confirm = parse_list(args.sweep_confirm_threshold, float)
    sweep_margin = parse_list(args.sweep_replace_margin, float)
    sweep_chunk = parse_list(args.sweep_target_chunk_len, int)

    new_cfgs = []
    for c, d, cf, m, ch in itertools.product(
        sweep_conf, sweep_draft, sweep_confirm, sweep_margin, sweep_chunk
    ):
        if d > cf:
            continue
        cfg = {
            "confidence_threshold": c,
            "draft_threshold": d,
            "confirm_threshold": cf,
            "replace_margin": m,
            "target_chunk_len": ch,
        }
        new_cfgs.append(cfg)

    total_units = len(seeds) + len(new_cfgs) * len(seeds)
    done_units = 0
    print_progress(done_units, total_units, prefix="Running")

    # Baseline once per seed, reused for all comparisons.
    base_rows = {}
    for seed in seeds:
        base_rows[seed] = run_one(module, model, seed, {**common, **baseline_cfg})
        done_units += 1
        print_progress(done_units, total_units, prefix="Running")

    run_rows = []
    agg_rows = []
    for cfg in new_cfgs:
        cid = config_id(cfg)
        per_seed = []
        for seed in seeds:
            r_new = run_one(module, model, seed, {**common, **cfg})
            r_base = base_rows[seed]
            row = {
                "config_id": cid,
                "seed": seed,
                **cfg,
                "base_elapsed_sec": r_base["elapsed_sec"],
                "new_elapsed_sec": r_new["elapsed_sec"],
                "base_total_steps": r_base["total_steps"],
                "new_total_steps": r_new["total_steps"],
                "base_avg_decoded_per_step": r_base["avg_decoded_per_step"],
                "new_avg_decoded_per_step": r_new["avg_decoded_per_step"],
                "base_distinct_2": r_base["distinct_2"],
                "new_distinct_2": r_new["distinct_2"],
                "base_repeat_char_ratio": r_base["repeat_char_ratio"],
                "new_repeat_char_ratio": r_new["repeat_char_ratio"],
                "base_char_entropy": r_base["char_entropy"],
                "new_char_entropy": r_new["char_entropy"],
            }
            run_rows.append(row)
            per_seed.append(row)
            done_units += 1
            print_progress(done_units, total_units, prefix="Running")

        agg = {
            "config_id": cid,
            **cfg,
            "n": len(per_seed),
            "delta_elapsed_sec_mean": mean(
                [r["new_elapsed_sec"] - r["base_elapsed_sec"] for r in per_seed]
            ),
            "delta_total_steps_mean": mean(
                [r["new_total_steps"] - r["base_total_steps"] for r in per_seed]
            ),
            "delta_avg_decoded_per_step_mean": mean(
                [
                    r["new_avg_decoded_per_step"] - r["base_avg_decoded_per_step"]
                    for r in per_seed
                ]
            ),
            "delta_distinct_2_mean": mean(
                [r["new_distinct_2"] - r["base_distinct_2"] for r in per_seed]
            ),
            "delta_repeat_char_ratio_mean": mean(
                [
                    r["new_repeat_char_ratio"] - r["base_repeat_char_ratio"]
                    for r in per_seed
                ]
            ),
            "delta_char_entropy_mean": mean(
                [r["new_char_entropy"] - r["base_char_entropy"] for r in per_seed]
            ),
        }
        agg_rows.append(agg)

    # Rank by speed first, then diversity.
    agg_rows.sort(
        key=lambda r: (
            r["delta_elapsed_sec_mean"],
            r["delta_total_steps_mean"],
            -r["delta_distinct_2_mean"],
            -r["delta_char_entropy_mean"],
        )
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.module}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_json = out_dir / "run_level.json"
    agg_json = out_dir / "agg_vs_baseline.json"
    summary_json = out_dir / "summary.json"
    fig_speed_quality = out_dir / "speed_vs_quality.png"
    fig_speed_parallel = out_dir / "speed_vs_parallel.png"
    fig_top_speed = out_dir / "top_speed_configs.png"

    run_json.write_text(json.dumps(run_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    agg_json.write_text(json.dumps(agg_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "config": vars(args),
        "baseline_cfg": baseline_cfg,
        "num_sweep_configs": len(new_cfgs),
        "top_5": agg_rows[:5],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_speed_vs_quality(agg_rows, fig_speed_quality)
    draw_speed_vs_parallel(agg_rows, fig_speed_parallel)
    draw_topk_speed_bars(agg_rows, fig_top_speed, top_k=8)

    print(f"Saved sweep report to: {out_dir}")
    print(f"- {run_json.name}")
    print(f"- {agg_json.name}")
    print(f"- {summary_json.name}")
    print(f"- {fig_speed_quality.name}")
    print(f"- {fig_speed_parallel.name}")
    print(f"- {fig_top_speed.name}")


if __name__ == "__main__":
    main()
