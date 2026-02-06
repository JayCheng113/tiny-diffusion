import argparse
import contextlib
import csv
import importlib
import io
import itertools
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a model checkpoint and run generation with different inference params."
    )
    parser.add_argument("--module", default="diffusion_flow", help="Python module name")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint")
    parser.add_argument("--seeds", default="1337", help="Comma-separated seeds")
    parser.add_argument("--max-new-tokens", default="512", help="Comma-separated ints")
    parser.add_argument("--prompt-len", default="16", help="Comma-separated ints")
    parser.add_argument("--temp", default="0.8", help="Comma-separated floats")
    parser.add_argument("--confidence-threshold", default="0.95", help="Comma-separated floats")
    parser.add_argument("--top-k", default="2", help="Comma-separated ints")
    parser.add_argument(
        "--flow-step",
        default="0.6",
        help="Comma-separated floats. Only used if generate() supports flow_step.",
    )
    parser.add_argument("--output-dir", default="infer_reports")
    return parser.parse_args()


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


def supports_flow_step(module):
    code_vars = module.generate.__code__.co_varnames
    return "flow_step" in code_vars


def main():
    args = parse_args()
    seeds = parse_list(args.seeds, int)
    max_new_tokens_list = parse_list(args.max_new_tokens, int)
    prompt_len_list = parse_list(args.prompt_len, int)
    temp_list = parse_list(args.temp, float)
    conf_list = parse_list(args.confidence_threshold, float)
    topk_list = parse_list(args.top_k, int)
    flow_step_list = parse_list(args.flow_step, float)

    module = importlib.import_module(args.module)
    device = module.device

    model = module.Model().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.module}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    txt_path = out_dir / "samples.txt"

    has_flow_step = supports_flow_step(module)

    rows = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv, open(
        txt_path, "w", encoding="utf-8"
    ) as f_txt:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "module",
                "weights",
                "seed",
                "max_new_tokens",
                "prompt_len",
                "temp",
                "confidence_threshold",
                "top_k",
                "flow_step",
                "elapsed_sec",
                "total_steps",
                "avg_decoded_per_step",
                "text_len",
            ]
        )

        if has_flow_step:
            grid_iter = itertools.product(
                seeds,
                max_new_tokens_list,
                prompt_len_list,
                temp_list,
                conf_list,
                topk_list,
                flow_step_list,
            )
        else:
            grid_iter = itertools.product(
                seeds,
                max_new_tokens_list,
                prompt_len_list,
                temp_list,
                conf_list,
                topk_list,
                [None],
            )

        run_id = 0
        for seed, max_new_tokens, prompt_len, temp, conf, top_k, flow_step in grid_iter:
            set_seed(seed)
            kwargs = dict(
                max_new_tokens=max_new_tokens,
                prompt_len=prompt_len,
                temp=temp,
                confidence_threshold=conf,
                top_k=top_k,
            )
            if has_flow_step and flow_step is not None:
                kwargs["flow_step"] = flow_step

            buf = io.StringIO()
            start = time.perf_counter()
            with contextlib.redirect_stdout(buf):
                text = module.generate(model, **kwargs)
            elapsed = time.perf_counter() - start
            total_steps, avg_decoded = parse_generate_stdout(buf.getvalue())

            row = {
                "module": args.module,
                "weights": os.path.abspath(args.weights),
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "prompt_len": prompt_len,
                "temp": temp,
                "confidence_threshold": conf,
                "top_k": top_k,
                "flow_step": flow_step,
                "elapsed_sec": elapsed,
                "total_steps": total_steps,
                "avg_decoded_per_step": avg_decoded,
                "text_len": len(text),
                "text": text,
            }
            rows.append(row)
            writer.writerow(
                [
                    row["module"],
                    row["weights"],
                    row["seed"],
                    row["max_new_tokens"],
                    row["prompt_len"],
                    row["temp"],
                    row["confidence_threshold"],
                    row["top_k"],
                    row["flow_step"],
                    f"{row['elapsed_sec']:.6f}",
                    row["total_steps"],
                    f"{row['avg_decoded_per_step']:.6f}",
                    row["text_len"],
                ]
            )
            f_csv.flush()

            f_txt.write(f"===== Run {run_id} =====\n")
            f_txt.write(json.dumps({k: v for k, v in row.items() if k != "text"}, ensure_ascii=False))
            f_txt.write("\n")
            f_txt.write(text)
            f_txt.write("\n\n")
            f_txt.flush()
            run_id += 1

    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(rows, f_json, ensure_ascii=False, indent=2)

    print(f"Saved sweep to: {out_dir}")
    print(f"- {csv_path.name}")
    print(f"- {json_path.name}")
    print(f"- {txt_path.name}")


if __name__ == "__main__":
    main()
