"""
Train and evaluate any decoder backbone + Engram with trust-weighted loss.

Quick-start (after installing requirements_neural.txt):

  # Train on synthetic corpus with GPT-2 medium (fast, no downloads)
  python run_engram_training.py --mode train --model gpt2-medium --corpus synthetic --steps 200

  # Train with Gemma 4 E4B-it (requires ~7 GB VRAM, downloads ~5 GB)
  python run_engram_training.py --mode train --corpus synthetic --steps 500

  # Train on real HF benchmarks
  python run_engram_training.py --mode train --corpus real --steps 1000

  # Evaluate a saved checkpoint
  python run_engram_training.py --mode eval --checkpoint checkpoints/checkpoint_step01000.pt

Paper alignment:
  - Default backbone: google/gemma-4-E4B-it (4-bit, ~5 GB VRAM) instead of DeepSeek 27B
  - Fallback: gpt2-medium for CPU/low-VRAM development
  - Engram injected after layer 1 (paper: layers 2 and 15 of a 30-layer model)
  - Trust weighting replaces uniform next-token loss weighting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from engram_backbone import EngramBackbone, GPT2WithEngram, build_canonical_map
from trust_trainer import TrainingConfig, train
from paper_benchmark_suite import build_cases, build_training_corpus
from gsm8k_eval import compare_engram_vs_baseline

ARTIFACT_DIR = Path("artifacts")
CHECKPOINT_DIR = Path("checkpoints")

_GEMMA_MODELS = {"google/gemma-4-E4B-it", "google/gemma-3-4b-it", "google/gemma-2-9b-it"}
_GPT2_MODELS = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    engram_layer: int = 1,
    force_cpu: bool = False,
) -> tuple[EngramBackbone, object]:
    """Load backbone + Engram. Quantizes automatically for Gemma-family models."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_quantized = (
        not force_cpu
        and torch.cuda.is_available()
        and model_name not in _GPT2_MODELS
    )

    if use_quantized:
        model = EngramBackbone.from_pretrained_quantized(
            model_name,
            engram_layer=engram_layer,
            max_ngram=3,
            num_hash_heads=4,
            num_buckets=16381,
        )
        # Engram module needs to be on CUDA for mixed-precision training
        device = next(model.backbone.parameters()).device
        model.engram = model.engram.to(device)
    else:
        # GPT-2 or CPU fallback
        if model_name not in _GPT2_MODELS:
            print(f"Warning: {model_name} is not a known GPT-2 variant and CUDA is unavailable.")
            print("Falling back to gpt2-medium.")
            model_name = "gpt2-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

        model = GPT2WithEngram.from_pretrained(
            model_name,
            engram_layer=engram_layer,
            max_ngram=3,
            num_hash_heads=4,
            num_buckets=16381,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    canonical_map = build_canonical_map(tokenizer)
    model.set_canonical_map(canonical_map)

    print(
        f"  backbone: {model.backbone_param_count():,} params  "
        f"engram: {model.engram_param_count():,} params  "
        f"trainable: {model.trainable_param_count():,} params"
    )
    return model, tokenizer


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    model, tokenizer = load_model_and_tokenizer(args.model, force_cpu=args.cpu)

    if args.full_corpus:
        from real_benchmark_suite import build_full_training_corpus
        cap = args.max_examples
        ds = set(args.datasets.split(",")) if args.datasets else None
        examples, _ = build_full_training_corpus(
            mmlu_limit=cap,
            gsm8k_limit=cap,
            triviaqa_limit=cap,
            popqa_limit=cap,
            datasets=ds,
        )
        print(f"Full corpus: {len(examples)} examples")
    elif args.corpus == "real":
        from real_benchmark_suite import build_real_benchmark_corpus
        examples, _ = build_real_benchmark_corpus(sample_limit=args.sample_limit)
        print(f"Real corpus: {len(examples)} examples (sample_limit={args.sample_limit})")
    else:
        examples = build_training_corpus()
        print(f"Synthetic corpus: {len(examples)} examples")

    config = TrainingConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        max_steps=args.steps,
        output_dir=args.output_dir,
        freeze_backbone=model.backbone_param_count() != model.trainable_param_count(),
    )

    log = train(model, examples, tokenizer, config)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    log_path = ARTIFACT_DIR / "engram_training_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Training log -> {log_path}")


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def _generate(model: EngramBackbone, tokenizer, prompt: str, max_new: int = 48) -> str:
    device = next(model.engram.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].size(1):], skip_special_tokens=True)


def run_eval(args: argparse.Namespace) -> None:
    model, tokenizer = load_model_and_tokenizer(args.model, force_cpu=args.cpu)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint (step {ckpt['step']})")

    from paper_benchmark_suite import run_suite
    proxy = run_suite()
    print("\n-- Symbolic proxy (engram_trust) --")
    print(json.dumps(proxy["models"], indent=2))

    model.eval()
    cases = build_cases()
    correct = 0
    results = []

    print("\n-- Neural Engram generation eval --")
    for case in cases:
        generated = _generate(model, tokenizer, case.query)
        hit = case.expected.lower() in generated.lower()
        correct += int(hit)
        results.append({
            "category": case.category,
            "query": case.query,
            "expected": case.expected,
            "generated": generated.strip(),
            "correct": hit,
        })
        mark = "+" if hit else "-"
        print(f"  [{mark}] {case.query[:55]:55s} -> {generated[:55]}")

    accuracy = correct / len(cases)
    print(f"\nNeural accuracy: {correct}/{len(cases)} = {accuracy:.1%}")

    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["correct"])
    for cat, hits in sorted(by_category.items()):
        print(f"  {cat}: {sum(hits)}/{len(hits)}")

    report = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "neural_accuracy": round(accuracy, 4),
        "by_category": {cat: round(sum(v)/len(v), 4) for cat, v in by_category.items()},
        "symbolic_proxy": proxy["models"],
        "results": results,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    report_path = ARTIFACT_DIR / "engram_neural_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport -> {report_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train / evaluate backbone + Engram")
    parser.add_argument("--mode", choices=["train", "eval", "eval_gsm8k"], default="train")
    parser.add_argument(
        "--model", default="google/gemma-4-E4B-it",
        help="HuggingFace model name. Gemma models load in 4-bit automatically. "
             "Use gpt2-medium for CPU/low-VRAM dev."
    )
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode (skips quantization, falls back to gpt2-medium)")
    parser.add_argument(
        "--corpus", choices=["synthetic", "real"], default="synthetic",
        help="Training corpus: 'synthetic' = paper-proxy; 'real' = HF datasets"
    )
    parser.add_argument("--full_corpus", action="store_true",
                        help="Use full public train splits (MMLU ~99K, GSM8K ~7.5K, TriviaQA ~87K, PopQA ~14K)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap rows per dataset when using --full_corpus (default: no cap)")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names to include with --full_corpus "
                             "(e.g. gsm8k or gsm8k,triviaqa). Default: all.")
    parser.add_argument("--sample_limit", type=int, default=64,
                        help="Rows per benchmark family for --corpus real")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Smaller default (2) for Gemma 4-bit VRAM budget")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for --mode eval / eval_gsm8k")
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="Number of GSM8K test questions for --mode eval_gsm8k")
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "eval_gsm8k":
        model, tokenizer = load_model_and_tokenizer(args.model, force_cpu=args.cpu)
        compare_engram_vs_baseline(
            model, tokenizer,
            checkpoint_path=args.checkpoint,
            n_samples=args.eval_samples,
        )
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
