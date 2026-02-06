import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SettingKey:
    module: str
    max_new_tokens: int
    prompt_len: int
    temp: float
    confidence_threshold: float
    top_k: int
    flow_step: float | None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze inference sweep outputs from run_inference_sweep.py"
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated list of results.json or results.csv files",
    )
    parser.add_argument("--output-dir", default="infer_analysis")
    parser.add_argument("--tag", default="analysis")
    return parser.parse_args()


def parse_float(v):
    if v is None or v == "" or v == "None":
        return None
    return float(v)


def text_metrics(text: str):
    n = len(text)
    if n == 0:
        return 0.0, 0.0, 0.0
    underscore_ratio = text.count("_") / n
    if n < 2:
        distinct2 = 0.0
        repeat_ratio = 0.0
    else:
        bigrams = [text[i : i + 2] for i in range(n - 1)]
        distinct2 = len(set(bigrams)) / len(bigrams)
        repeats = sum(1 for i in range(1, n) if text[i] == text[i - 1])
        repeat_ratio = repeats / (n - 1)
    return distinct2, repeat_ratio, underscore_ratio


def load_rows(path: Path):
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for r in rows:
            text = r.get("text", "")
            distinct2, repeat_ratio, underscore_ratio = text_metrics(text)
            out.append(
                {
                    "module": r["module"],
                    "max_new_tokens": int(r["max_new_tokens"]),
                    "prompt_len": int(r["prompt_len"]),
                    "temp": float(r["temp"]),
                    "confidence_threshold": float(r["confidence_threshold"]),
                    "top_k": int(r["top_k"]),
                    "flow_step": parse_float(r.get("flow_step")),
                    "elapsed_sec": float(r["elapsed_sec"]),
                    "total_steps": float(r["total_steps"]),
                    "avg_decoded_per_step": float(r["avg_decoded_per_step"]),
                    "text_len": float(r["text_len"]),
                    "distinct2": distinct2,
                    "repeat_ratio": repeat_ratio,
                    "underscore_ratio": underscore_ratio,
                }
            )
        return out

    if path.suffix.lower() == ".csv":
        out = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                out.append(
                    {
                        "module": r["module"],
                        "max_new_tokens": int(r["max_new_tokens"]),
                        "prompt_len": int(r["prompt_len"]),
                        "temp": float(r["temp"]),
                        "confidence_threshold": float(r["confidence_threshold"]),
                        "top_k": int(r["top_k"]),
                        "flow_step": parse_float(r.get("flow_step")),
                        "elapsed_sec": float(r["elapsed_sec"]),
                        "total_steps": float(r["total_steps"]),
                        "avg_decoded_per_step": float(r["avg_decoded_per_step"]),
                        "text_len": float(r["text_len"]),
                        "distinct2": None,
                        "repeat_ratio": None,
                        "underscore_ratio": None,
                    }
                )
        return out

    raise ValueError(f"Unsupported file type: {path}")


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def std(vals):
    if not vals or len(vals) == 1:
        return 0.0
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def aggregate(rows):
    buckets = defaultdict(list)
    for r in rows:
        key = SettingKey(
            module=r["module"],
            max_new_tokens=r["max_new_tokens"],
            prompt_len=r["prompt_len"],
            temp=r["temp"],
            confidence_threshold=r["confidence_threshold"],
            top_k=r["top_k"],
            flow_step=r["flow_step"],
        )
        buckets[key].append(r)

    agg_rows = []
    for key, rs in buckets.items():
        elapsed = [x["elapsed_sec"] for x in rs]
        steps = [x["total_steps"] for x in rs]
        decoded = [x["avg_decoded_per_step"] for x in rs]
        text_len = [x["text_len"] for x in rs]
        d2 = [x["distinct2"] for x in rs if x["distinct2"] is not None]
        rep = [x["repeat_ratio"] for x in rs if x["repeat_ratio"] is not None]
        und = [x["underscore_ratio"] for x in rs if x["underscore_ratio"] is not None]

        quality_score = None
        if d2 and rep:
            quality_score = mean(d2) - mean(rep)

        agg_rows.append(
            {
                "module": key.module,
                "max_new_tokens": key.max_new_tokens,
                "prompt_len": key.prompt_len,
                "temp": key.temp,
                "confidence_threshold": key.confidence_threshold,
                "top_k": key.top_k,
                "flow_step": key.flow_step,
                "n": len(rs),
                "elapsed_sec_mean": mean(elapsed),
                "elapsed_sec_std": std(elapsed),
                "total_steps_mean": mean(steps),
                "avg_decoded_per_step_mean": mean(decoded),
                "text_len_mean": mean(text_len),
                "distinct2_mean": mean(d2) if d2 else None,
                "repeat_ratio_mean": mean(rep) if rep else None,
                "underscore_ratio_mean": mean(und) if und else None,
                "quality_score": quality_score,
            }
        )
    return agg_rows


def pareto_front(rows, x_key, y_key, minimize_x=True, maximize_y=True):
    points = []
    for i, r in enumerate(rows):
        xv = r.get(x_key)
        yv = r.get(y_key)
        if xv is None or yv is None:
            continue
        points.append((i, xv, yv))

    keep = []
    for i, x_i, y_i in points:
        dominated = False
        for j, x_j, y_j in points:
            if i == j:
                continue
            better_x = x_j <= x_i if minimize_x else x_j >= x_i
            better_y = y_j >= y_i if maximize_y else y_j <= y_i
            strict_x = x_j < x_i if minimize_x else x_j > x_i
            strict_y = y_j > y_i if maximize_y else y_j < y_i
            if better_x and better_y and (strict_x or strict_y):
                dominated = True
                break
        if not dominated:
            keep.append(rows[i])
    return keep


def save_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def scatter_plot(rows, x_key, y_key, out_path, title):
    plt.figure(figsize=(8, 5))
    modules = sorted(set(r["module"] for r in rows))
    for mod in modules:
        xs = [r[x_key] for r in rows if r["module"] == mod and r.get(x_key) is not None]
        ys = [r[y_key] for r in rows if r["module"] == mod and r.get(y_key) is not None]
        if xs and ys:
            plt.scatter(xs, ys, label=mod, alpha=0.8)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    input_paths = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    all_rows = []
    for p in input_paths:
        all_rows.extend(load_rows(p))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_rows = aggregate(all_rows)
    agg_rows = sorted(
        agg_rows,
        key=lambda r: (r["elapsed_sec_mean"], -r["avg_decoded_per_step_mean"]),
    )

    pareto_speed_parallel = pareto_front(
        agg_rows, "elapsed_sec_mean", "avg_decoded_per_step_mean", minimize_x=True, maximize_y=True
    )
    pareto_speed_quality = pareto_front(
        agg_rows, "elapsed_sec_mean", "quality_score", minimize_x=True, maximize_y=True
    )

    aggregated_csv = out_dir / f"{args.tag}_aggregated.csv"
    pareto_parallel_csv = out_dir / f"{args.tag}_pareto_speed_parallel.csv"
    pareto_quality_csv = out_dir / f"{args.tag}_pareto_speed_quality.csv"
    summary_json = out_dir / f"{args.tag}_summary.json"

    save_csv(aggregated_csv, agg_rows)
    save_csv(pareto_parallel_csv, pareto_speed_parallel)
    save_csv(pareto_quality_csv, pareto_speed_quality)

    top_fast = sorted(agg_rows, key=lambda r: r["elapsed_sec_mean"])[:5]
    top_parallel = sorted(agg_rows, key=lambda r: -r["avg_decoded_per_step_mean"])[:5]
    top_quality = sorted(
        [r for r in agg_rows if r["quality_score"] is not None],
        key=lambda r: -r["quality_score"],
    )[:5]

    summary = {
        "num_inputs": len(input_paths),
        "num_runs": len(all_rows),
        "num_settings": len(agg_rows),
        "top_fastest": top_fast,
        "top_parallel": top_parallel,
        "top_quality": top_quality,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    scatter_plot(
        agg_rows,
        "elapsed_sec_mean",
        "avg_decoded_per_step_mean",
        out_dir / f"{args.tag}_speed_vs_parallel.png",
        "Speed vs Parallel Decode",
    )

    has_quality = any(r.get("quality_score") is not None for r in agg_rows)
    if has_quality:
        scatter_plot(
            [r for r in agg_rows if r.get("quality_score") is not None],
            "elapsed_sec_mean",
            "quality_score",
            out_dir / f"{args.tag}_speed_vs_quality.png",
            "Speed vs Quality Score (distinct2 - repeat)",
        )

    print(f"Saved analysis to: {out_dir}")
    print(f"- {aggregated_csv.name}")
    print(f"- {pareto_parallel_csv.name}")
    print(f"- {pareto_quality_csv.name}")
    print(f"- {summary_json.name}")
    print(f"- {args.tag}_speed_vs_parallel.png")
    if has_quality:
        print(f"- {args.tag}_speed_vs_quality.png")


if __name__ == "__main__":
    main()
