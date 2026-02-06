import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze text quality from evaluate_generate_methods results.json."
    )
    parser.add_argument("--input", required=True, help="Path to results.json")
    parser.add_argument(
        "--output-dir",
        default="eval_generate_quality",
        help="Directory to save quality reports",
    )
    parser.add_argument(
        "--tag",
        default="quality",
        help="Output file tag",
    )
    return parser.parse_args()


def text_metrics(text: str):
    n = len(text)
    if n == 0:
        return {
            "text_len": 0,
            "underscore_ratio": 0.0,
            "distinct_2": 0.0,
            "distinct_3": 0.0,
            "repeat_char_ratio": 0.0,
            "char_entropy": 0.0,
        }

    underscore_ratio = text.count("_") / n

    if n >= 2:
        grams2 = [text[i : i + 2] for i in range(n - 1)]
        distinct_2 = len(set(grams2)) / len(grams2)
    else:
        distinct_2 = 0.0

    if n >= 3:
        grams3 = [text[i : i + 3] for i in range(n - 2)]
        distinct_3 = len(set(grams3)) / len(grams3)
    else:
        distinct_3 = 0.0

    repeats = sum(1 for i in range(1, n) if text[i] == text[i - 1])
    repeat_char_ratio = repeats / max(n - 1, 1)

    cnt = Counter(text)
    entropy = 0.0
    for c in cnt.values():
        p = c / n
        entropy -= p * math.log(p, 2)

    return {
        "text_len": n,
        "underscore_ratio": float(underscore_ratio),
        "distinct_2": float(distinct_2),
        "distinct_3": float(distinct_3),
        "repeat_char_ratio": float(repeat_char_ratio),
        "char_entropy": float(entropy),
    }


def aggregate(rows):
    def ms(values):
        return {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }

    keys = [
        "text_len",
        "underscore_ratio",
        "distinct_2",
        "distinct_3",
        "repeat_char_ratio",
        "char_entropy",
    ]
    out = {k: ms([r[k] for r in rows]) for k in keys}
    out["n"] = len(rows)
    return out


def compare_pairwise(rows):
    # Compare two methods by seed using a simple score:
    # better if high: distinct_2, distinct_3, char_entropy
    # better if low: underscore_ratio, repeat_char_ratio
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["method"]] = r

    methods = sorted({r["method"] for r in rows})
    if len(methods) != 2:
        return {
            "note": "pairwise comparison currently supports exactly 2 methods",
            "methods": methods,
        }

    a, b = methods
    paired = []
    for seed, m in by_seed.items():
        if a in m and b in m:
            paired.append((seed, m[a], m[b]))

    a_wins = 0
    b_wins = 0
    ties = 0
    details = []
    for seed, ra, rb in paired:
        score_a = 0
        score_b = 0
        for k in ("distinct_2", "distinct_3", "char_entropy"):
            if ra[k] > rb[k]:
                score_a += 1
            elif rb[k] > ra[k]:
                score_b += 1
        for k in ("underscore_ratio", "repeat_char_ratio"):
            if ra[k] < rb[k]:
                score_a += 1
            elif rb[k] < ra[k]:
                score_b += 1

        if score_a > score_b:
            a_wins += 1
            winner = a
        elif score_b > score_a:
            b_wins += 1
            winner = b
        else:
            ties += 1
            winner = "tie"

        details.append(
            {
                "seed": seed,
                "winner": winner,
                "score_a": score_a,
                "score_b": score_b,
            }
        )

    return {
        "methods": [a, b],
        "paired_n": len(paired),
        "wins": {a: a_wins, b: b_wins, "tie": ties},
        "details": details,
    }


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {args.input}")

    rows = json.loads(in_path.read_text(encoding="utf-8"))
    enriched = []
    for r in rows:
        tm = text_metrics(r.get("text", ""))
        enriched.append(
            {
                "method": r["method"],
                "seed": r["seed"],
                **tm,
            }
        )

    methods = sorted({r["method"] for r in enriched})
    by_method = {m: [r for r in enriched if r["method"] == m] for m in methods}
    summary_by_method = {m: aggregate(rs) for m, rs in by_method.items()}
    pairwise = compare_pairwise(enriched)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.tag}_summary.json"
    rows_path = out_dir / f"{args.tag}_rows.json"

    summary = {
        "input": str(in_path.resolve()),
        "methods": methods,
        "quality_summary": summary_by_method,
        "pairwise": pairwise,
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved quality summary: {summary_path}")
    print(f"Saved quality rows: {rows_path}")


if __name__ == "__main__":
    main()
