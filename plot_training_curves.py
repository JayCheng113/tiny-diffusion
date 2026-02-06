import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves from CSV logs.")
    parser.add_argument(
        "--baseline-log",
        required=True,
        help="Path to baseline log CSV (from diffusion.py).",
    )
    parser.add_argument(
        "--flow-log",
        required=True,
        help="Path to flow log CSV (from diffusion_flow.py).",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--tag",
        default="comparison",
        help="Filename prefix for output figures.",
    )
    return parser.parse_args()


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = None
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def col(rows, key):
    return [r[key] for r in rows if key in r and r[key] is not None]


def paired_xy(rows, x_key, y_key):
    xs, ys = [], []
    for r in rows:
        if x_key in r and y_key in r and r[x_key] is not None and r[y_key] is not None:
            xs.append(r[x_key])
            ys.append(r[y_key])
    return xs, ys


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_train_val_loss(baseline_rows, flow_rows, out_path):
    plt.figure(figsize=(9, 5))
    bx, by = paired_xy(baseline_rows, "step", "train_loss")
    fx, fy = paired_xy(flow_rows, "step", "train_loss")
    plt.plot(bx, by, label="baseline train_loss", linewidth=2)
    plt.plot(fx, fy, label="flow train_loss", linewidth=2)

    bx, by = paired_xy(baseline_rows, "step", "val_loss")
    fx, fy = paired_xy(flow_rows, "step", "val_loss")
    plt.plot(bx, by, label="baseline val_loss", linestyle="--", linewidth=2)
    plt.plot(fx, fy, label="flow val_loss", linestyle="--", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_batch_total_loss(baseline_rows, flow_rows, out_path):
    plt.figure(figsize=(9, 5))
    bx, by = paired_xy(baseline_rows, "step", "batch_total_loss")
    fx, fy = paired_xy(flow_rows, "step", "batch_total_loss")
    plt.plot(bx, by, label="baseline batch_total_loss", linewidth=2)
    plt.plot(fx, fy, label="flow batch_total_loss", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Batch Total Loss")
    plt.title("Batch Total Loss Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_flow_breakdown(flow_rows, out_path):
    plt.figure(figsize=(9, 5))
    sx, sy = paired_xy(flow_rows, "step", "batch_ce_loss")
    plt.plot(sx, sy, label="flow batch_ce_loss", linewidth=2)

    sx, sy = paired_xy(flow_rows, "step", "batch_flow_loss")
    if len(sx) > 0:
        plt.plot(sx, sy, label="flow batch_flow_loss", linewidth=2)

    sx, sy = paired_xy(flow_rows, "step", "batch_total_loss")
    plt.plot(sx, sy, label="flow batch_total_loss", linestyle="--", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Flow Model Loss Breakdown")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_wallclock_progress(baseline_rows, flow_rows, out_path):
    plt.figure(figsize=(9, 5))
    bx, by = paired_xy(baseline_rows, "elapsed_sec", "val_loss")
    fx, fy = paired_xy(flow_rows, "elapsed_sec", "val_loss")
    plt.plot(bx, by, label="baseline val_loss vs time", linewidth=2)
    plt.plot(fx, fy, label="flow val_loss vs time", linewidth=2)

    plt.xlabel("Elapsed Seconds")
    plt.ylabel("Val Loss")
    plt.title("Validation Loss vs Wall-Clock Time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    baseline_rows = read_csv(args.baseline_log)
    flow_rows = read_csv(args.flow_log)

    plot_train_val_loss(
        baseline_rows,
        flow_rows,
        out_dir / f"{args.tag}_train_val_loss.png",
    )
    plot_batch_total_loss(
        baseline_rows,
        flow_rows,
        out_dir / f"{args.tag}_batch_total_loss.png",
    )
    plot_flow_breakdown(
        flow_rows,
        out_dir / f"{args.tag}_flow_breakdown.png",
    )
    plot_wallclock_progress(
        baseline_rows,
        flow_rows,
        out_dir / f"{args.tag}_val_vs_time.png",
    )

    print(f"Saved plots to: {out_dir}")
    print(f"- {args.tag}_train_val_loss.png")
    print(f"- {args.tag}_batch_total_loss.png")
    print(f"- {args.tag}_flow_breakdown.png")
    print(f"- {args.tag}_val_vs_time.png")


if __name__ == "__main__":
    main()
