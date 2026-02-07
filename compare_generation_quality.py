import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

import diffusion as rope_mod
import diffusion_polar as polar_mod


def parse_prompt_starts(data_len, prompt_len, num_prompts, seed):
    if data_len <= prompt_len + 1:
        raise ValueError("Dataset is too short for the requested prompt length.")
    g = torch.Generator()
    g.manual_seed(seed)
    high = data_len - prompt_len - 1
    starts = torch.randint(0, high, (num_prompts,), generator=g).tolist()
    return starts


def distinct_n(token_ids, n):
    total = len(token_ids) - n + 1
    if total <= 0:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(total)]
    return len(set(ngrams)) / total


def repeat_ratio_n(token_ids, n):
    total = len(token_ids) - n + 1
    if total <= 0:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(total)]
    return 1.0 - (len(set(ngrams)) / total)


@torch.no_grad()
def generate_with_stats(
    module,
    model,
    prompt_tokens,
    max_new_tokens,
    temp=1.0,
    confidence_threshold=0.95,
    top_k=3,
    sample_seed=0,
):
    device = module.device
    prompt_len = len(prompt_tokens)
    all_tokens = list(prompt_tokens)
    total_steps = 0
    masked_confidences = []

    torch.manual_seed(sample_seed)

    while len(all_tokens) - prompt_len < max_new_tokens:
        remaining = max_new_tokens - (len(all_tokens) - prompt_len)
        block_len = min(240, remaining)

        x = torch.full(
            (1, module.block_size),
            module.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        x[0, :prompt_len] = torch.tensor(
            all_tokens[-prompt_len:], dtype=torch.long, device=device
        )

        masked = torch.zeros(1, module.block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        while masked.any():
            total_steps += 1
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)
            masked_confidences.append(confidences[masked].mean().item())

            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_scores = torch.where(
                    masked, confidences, torch.tensor(-float("inf"), device=device)
                )
                decode_mask.view(-1)[masked_scores.argmax()] = True

            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(
                1, module.block_size
            )
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

        all_tokens.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    generated_tokens = all_tokens[prompt_len : prompt_len + max_new_tokens]
    text = module.decode(prompt_tokens + generated_tokens)
    avg_conf = statistics.fmean(masked_confidences) if masked_confidences else 0.0
    avg_decoded_per_step = (
        len(generated_tokens) / total_steps if total_steps > 0 else float("inf")
    )
    return {
        "generated_tokens": generated_tokens,
        "full_text": text,
        "steps": total_steps,
        "avg_decoded_per_step": avg_decoded_per_step,
        "avg_masked_confidence": avg_conf,
    }


def load_model(module, weights_path):
    model = module.Model().to(module.device)
    state_dict = torch.load(weights_path, map_location=module.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def summarize(metric_list):
    if not metric_list:
        return {"mean": 0.0, "std": 0.0}
    mean = statistics.fmean(metric_list)
    std = statistics.pstdev(metric_list) if len(metric_list) > 1 else 0.0
    return {"mean": mean, "std": std}


def main():
    parser = argparse.ArgumentParser(description="Compare generated text quality for RoPE vs Polar.")
    parser.add_argument("--rope-weights", required=True)
    parser.add_argument("--polar-weights", required=True)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="generation_eval.json")
    args = parser.parse_args()

    rope_model = load_model(rope_mod, args.rope_weights)
    polar_model = load_model(polar_mod, args.polar_weights)

    starts = parse_prompt_starts(
        data_len=len(rope_mod.val_data),
        prompt_len=args.prompt_len,
        num_prompts=args.num_prompts,
        seed=args.seed,
    )

    variants = {
        "rope": {"module": rope_mod, "model": rope_model, "results": []},
        "polar": {"module": polar_mod, "model": polar_model, "results": []},
    }

    for i, start in enumerate(starts):
        prompt_tokens = rope_mod.val_data[start : start + args.prompt_len].tolist()
        prompt_text = rope_mod.decode(prompt_tokens)
        print(f"\nPrompt {i + 1}/{len(starts)}")
        print(f"Prompt: {repr(prompt_text)}")
        for variant_name, payload in variants.items():
            out = generate_with_stats(
                module=payload["module"],
                model=payload["model"],
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.gen_len,
                temp=args.temp,
                confidence_threshold=args.confidence_threshold,
                top_k=args.top_k,
                sample_seed=args.seed + i,
            )
            gen_ids = out["generated_tokens"]
            metrics = {
                "distinct_1": distinct_n(gen_ids, 1),
                "distinct_2": distinct_n(gen_ids, 2),
                "repeat_3gram_ratio": repeat_ratio_n(gen_ids, 3),
                "avg_decoded_per_step": out["avg_decoded_per_step"],
                "avg_masked_confidence": out["avg_masked_confidence"],
            }
            payload["results"].append(
                {
                    "prompt_start": int(start),
                    "prompt_text": prompt_text,
                    "generated_text": payload["module"].decode(gen_ids),
                    "metrics": metrics,
                }
            )
            print(
                f"{variant_name}: repeat_3gram={metrics['repeat_3gram_ratio']:.4f}, "
                f"distinct_2={metrics['distinct_2']:.4f}, "
                f"decoded/step={metrics['avg_decoded_per_step']:.2f}"
            )

    summary = {}
    for variant_name, payload in variants.items():
        results = payload["results"]
        summary[variant_name] = {
            "repeat_3gram_ratio": summarize(
                [r["metrics"]["repeat_3gram_ratio"] for r in results]
            ),
            "distinct_1": summarize([r["metrics"]["distinct_1"] for r in results]),
            "distinct_2": summarize([r["metrics"]["distinct_2"] for r in results]),
            "avg_decoded_per_step": summarize(
                [r["metrics"]["avg_decoded_per_step"] for r in results]
            ),
            "avg_masked_confidence": summarize(
                [r["metrics"]["avg_masked_confidence"] for r in results]
            ),
        }

    output = {
        "config": {
            "num_prompts": args.num_prompts,
            "prompt_len": args.prompt_len,
            "gen_len": args.gen_len,
            "temp": args.temp,
            "top_k": args.top_k,
            "confidence_threshold": args.confidence_threshold,
            "seed": args.seed,
            "rope_weights": args.rope_weights,
            "polar_weights": args.polar_weights,
        },
        "summary": summary,
        "runs": {k: v["results"] for k, v in variants.items()},
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== Generation Summary ===")
    for variant_name in ["rope", "polar"]:
        s = summary[variant_name]
        print(
            f"{variant_name}: repeat_3gram={s['repeat_3gram_ratio']['mean']:.4f}±{s['repeat_3gram_ratio']['std']:.4f}, "
            f"distinct_2={s['distinct_2']['mean']:.4f}±{s['distinct_2']['std']:.4f}, "
            f"decoded/step={s['avg_decoded_per_step']['mean']:.2f}±{s['avg_decoded_per_step']['std']:.2f}"
        )
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
