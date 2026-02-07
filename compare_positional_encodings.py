import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

VARIANTS = {
    "rope": "diffusion.py",
    "polar": "diffusion_polar.py",
}

METRIC_RE = re.compile(r"METRIC step=(\d+) train_loss=([0-9.]+) val_loss=([0-9.]+)")
BEST_RE = re.compile(r"METRIC best_val_loss=([0-9.]+)")


def parse_seeds(seed_text):
    return [int(s.strip()) for s in seed_text.split(",") if s.strip()]


def run_one(repo_dir, python_cmd, variant, seed, args):
    script = VARIANTS[variant]
    env = os.environ.copy()
    env.update(
        {
            "TD_TRAIN": "1",
            "TD_LOAD_WEIGHTS": "0",
            "TD_SAVE_WEIGHTS": "1" if args.save_weights else "0",
            "TD_SAMPLE_DURING_TRAIN": "0",
            "TD_RUN_GENERATE": "0",
            "TD_SEED": str(seed),
            "TD_MAX_ITERS": str(args.max_iters),
            "TD_EVAL_INTERVAL": str(args.eval_interval),
            "TD_EVAL_ITERS": str(args.eval_iters),
            "TD_BATCH_SIZE": str(args.batch_size),
            "TD_BLOCK_SIZE": str(args.block_size),
            "TD_LEARNING_RATE": str(args.learning_rate),
            "TD_WEIGHTS_PATH": str(repo_dir / "weights" / "ablation" / f"{variant}_seed{seed}.pt"),
        }
    )

    cmd = [python_cmd, script, "--train"]
    print(f"\\n=== Running {variant} seed={seed} ===")

    process = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    eval_points = []
    best_val_loss = None
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(f"[{variant}:{seed}] {line}")
        metric_match = METRIC_RE.search(line)
        if metric_match:
            step, train_loss, val_loss = metric_match.groups()
            eval_points.append(
                {
                    "step": int(step),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )
        best_match = BEST_RE.search(line)
        if best_match:
            best_val_loss = float(best_match.group(1))

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{script} failed for seed={seed} with return code {return_code}")
    if not eval_points:
        raise RuntimeError(f"No METRIC lines parsed for {script} seed={seed}")

    final_val_loss = eval_points[-1]["val_loss"]
    if best_val_loss is None:
        best_val_loss = min(point["val_loss"] for point in eval_points)

    return {
        "variant": variant,
        "seed": seed,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "eval_points": eval_points,
    }


def summarize(records):
    by_variant = {}
    for record in records:
        by_variant.setdefault(record["variant"], []).append(record)

    summary = {}
    for variant, items in by_variant.items():
        best_vals = [x["best_val_loss"] for x in items]
        final_vals = [x["final_val_loss"] for x in items]
        summary[variant] = {
            "runs": len(items),
            "best_val_loss_mean": statistics.fmean(best_vals),
            "best_val_loss_std": statistics.pstdev(best_vals) if len(best_vals) > 1 else 0.0,
            "final_val_loss_mean": statistics.fmean(final_vals),
            "final_val_loss_std": statistics.pstdev(final_vals) if len(final_vals) > 1 else 0.0,
        }
    return summary


def make_plots(records, summary, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or use --no-plot."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    by_variant = {}
    for record in records:
        by_variant.setdefault(record["variant"], []).append(record)

    # Plot 1: mean +/- std validation curve over steps
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"rope": "#1f77b4", "polar": "#d62728"}
    for variant in ["rope", "polar"]:
        runs = by_variant.get(variant, [])
        if not runs:
            continue
        steps = [p["step"] for p in runs[0]["eval_points"]]
        val_matrix = [[p["val_loss"] for p in run["eval_points"]] for run in runs]
        means = [statistics.fmean(col) for col in zip(*val_matrix)]
        stds = [
            statistics.pstdev(col) if len(col) > 1 else 0.0 for col in zip(*val_matrix)
        ]
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.plot(steps, means, label=f"{variant} mean", color=colors[variant], linewidth=2)
        ax.fill_between(steps, lower, upper, color=colors[variant], alpha=0.2, linewidth=0)

    ax.set_title("Validation Loss vs Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    curve_path = output_dir / "val_loss_curve.png"
    fig.tight_layout()
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)

    # Plot 2: best/final val loss bar chart with std error bars
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = ["rope", "polar"]
    x = list(range(len(variants)))
    width = 0.36

    best_means = [summary[v]["best_val_loss_mean"] for v in variants]
    best_stds = [summary[v]["best_val_loss_std"] for v in variants]
    final_means = [summary[v]["final_val_loss_mean"] for v in variants]
    final_stds = [summary[v]["final_val_loss_std"] for v in variants]

    ax.bar(
        [i - width / 2 for i in x],
        best_means,
        width=width,
        yerr=best_stds,
        capsize=4,
        label="best_val_loss",
        color="#2ca02c",
    )
    ax.bar(
        [i + width / 2 for i in x],
        final_means,
        width=width,
        yerr=final_stds,
        capsize=4,
        label="final_val_loss",
        color="#ff7f0e",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Loss")
    ax.set_title("A/B Summary (mean ± std across seeds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    bar_path = output_dir / "summary_bars.png"
    fig.tight_layout()
    fig.savefig(bar_path, dpi=160)
    plt.close(fig)

    return {"val_loss_curve": str(curve_path), "summary_bars": str(bar_path)}


def main():
    parser = argparse.ArgumentParser(description="A/B compare RoPE vs Polar positional encoding")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run training scripts")
    parser.add_argument("--seeds", default="1337,2027,9001", help="Comma-separated random seeds")
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", default="ablation_results.json")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-dir", default="ablation_plots")
    parser.add_argument(
        "--no-save-weights",
        dest="save_weights",
        action="store_false",
        help="Disable saving per-run checkpoints under weights/ablation.",
    )
    parser.set_defaults(save_weights=True)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    seeds = parse_seeds(args.seeds)

    all_records = []
    for variant in ["rope", "polar"]:
        for seed in seeds:
            all_records.append(run_one(repo_dir, args.python, variant, seed, args))

    summary = summarize(all_records)
    result = {
        "config": {
            "seeds": seeds,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "learning_rate": args.learning_rate,
            "save_weights": args.save_weights,
        },
        "summary": summary,
        "runs": all_records,
    }

    if not args.no_plot:
        plot_paths = make_plots(all_records, summary, repo_dir / args.plot_dir)
        result["plots"] = plot_paths

    output_path = repo_dir / args.output
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\\n=== Summary ===")
    for variant in ["rope", "polar"]:
        s = summary[variant]
        print(
            f"{variant}: best_val={s['best_val_loss_mean']:.4f}±{s['best_val_loss_std']:.4f}, "
            f"final_val={s['final_val_loss_mean']:.4f}±{s['final_val_loss_std']:.4f}, runs={s['runs']}"
        )
    if not args.no_plot:
        print(f"Plots: {result['plots']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
