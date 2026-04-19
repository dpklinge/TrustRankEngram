"""
GSM8K evaluation for EngramBackbone models.

Runs two passes over the same test questions:
  1. Engram ENABLED  — backbone + Engram module (trained checkpoint)
  2. Engram DISABLED — same backbone, hook bypassed (pure Gemma baseline)

Answer extraction: looks for the GSM8K "####" marker, then falls back to
the last integer/decimal in the generated text.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import torch

from engram_backbone import EngramBackbone
from real_benchmark_suite import take_rows, extract_gsm8k_answer, normalize_answer

ARTIFACT_DIR = Path("artifacts")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


# ------------------------------------------------------------------
# Prompt formatting
# ------------------------------------------------------------------

def _format_prompt(question: str, tokenizer) -> str:
    """Apply the model's chat template for instruction-following."""
    messages = [{"role": "user", "content": (
        f"{question}\n\nSolve step by step. "
        "At the end write the final numeric answer after ####."
    )}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without a chat template
        return f"Question: {question}\nAnswer step by step. Final answer after ####:\n"


# ------------------------------------------------------------------
# Answer extraction
# ------------------------------------------------------------------

def extract_generated_answer(text: str) -> str:
    """Extract the numeric answer from generated text."""
    # Prefer the GSM8K #### marker
    if "####" in text:
        after = text.split("####")[-1].strip()
        nums = _NUMBER_RE.findall(after)
        if nums:
            return nums[0].replace(",", "")
    # Fall back to last number in text
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return text.strip()


def answers_match(predicted: str, expected: str) -> bool:
    pred = normalize_answer(predicted.replace(",", ""))
    exp = normalize_answer(expected.replace(",", ""))
    return pred == exp


# ------------------------------------------------------------------
# Single-pass generation
# ------------------------------------------------------------------

def _generate_answer(
    model: EngramBackbone,
    tokenizer,
    question: str,
    max_new_tokens: int = 128,
) -> str:
    device = next(model.engram.parameters()).device
    prompt = _format_prompt(question, tokenizer)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).to(device)
    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][enc["input_ids"].size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ------------------------------------------------------------------
# Full eval pass
# ------------------------------------------------------------------

def run_gsm8k_eval(
    model: EngramBackbone,
    tokenizer,
    n_samples: int = 200,
    engram_enabled: bool = True,
    verbose: bool = True,
) -> dict:
    """Evaluate on n_samples GSM8K test questions.

    Returns a dict with accuracy, per-example results, and timing.
    """
    label = "engram" if engram_enabled else "baseline"
    model.engram_enabled = engram_enabled

    rows = take_rows("gsm8k", "main", "test", n_samples)
    correct = 0
    results = []
    t0 = time.time()

    for i, row in enumerate(rows):
        torch.cuda.empty_cache()
        question = str(row["question"])
        expected = extract_gsm8k_answer(str(row["answer"]))
        generated = _generate_answer(model, tokenizer, question)
        predicted = extract_generated_answer(generated)
        hit = answers_match(predicted, expected)
        correct += int(hit)
        results.append({
            "id": i,
            "question": question[:120],
            "expected": expected,
            "predicted": predicted,
            "generated": generated[:300],
            "correct": hit,
        })
        if verbose and i == 0:
            print(f"  [sample] Q: {question[:80]}")
            print(f"  [sample] generated: {generated[:200]}")
            print(f"  [sample] predicted={predicted!r}  expected={expected!r}  hit={hit}")
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            acc = correct / (i + 1)
            print(
                f"  [{label}] {i+1:3d}/{n_samples}  "
                f"acc={acc:.1%}  elapsed={elapsed/60:.1f}min",
                flush=True,
            )

    accuracy = correct / len(rows) if rows else 0.0
    elapsed = time.time() - t0
    return {
        "label": label,
        "n_samples": len(rows),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }


# ------------------------------------------------------------------
# Compare Engram vs Baseline on the same questions
# ------------------------------------------------------------------

def compare_engram_vs_baseline(
    model: EngramBackbone,
    tokenizer,
    checkpoint_path: Optional[str],
    n_samples: int = 200,
) -> dict:
    """Run both passes and save a combined report."""

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        step = ckpt.get("step", "?")
        # Only restore Engram weights — backbone is already loaded from HuggingFace
        engram_state = {
            k[len("engram."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("engram.")
        }
        model.engram.load_state_dict(engram_state)
        print(f"Loaded Engram weights from checkpoint step {step}")
    else:
        step = "untrained"

    model.eval()
    torch.cuda.empty_cache()

    print(f"\n=== Pass 1/2: Gemma + Engram (trained) ===")
    engram_result = run_gsm8k_eval(model, tokenizer, n_samples=n_samples, engram_enabled=True)
    print(f"  Engram accuracy: {engram_result['correct']}/{engram_result['n_samples']} = {engram_result['accuracy']:.1%}")

    torch.cuda.empty_cache()
    print(f"\n=== Pass 2/2: Gemma baseline (Engram disabled) ===")
    baseline_result = run_gsm8k_eval(model, tokenizer, n_samples=n_samples, engram_enabled=False)
    print(f"  Baseline accuracy: {baseline_result['correct']}/{baseline_result['n_samples']} = {baseline_result['accuracy']:.1%}")

    delta = engram_result["accuracy"] - baseline_result["accuracy"]
    print(f"\n=== Summary ===")
    print(f"  Baseline (Gemma alone):  {baseline_result['accuracy']:.1%}")
    print(f"  Engram (Gemma + module): {engram_result['accuracy']:.1%}")
    print(f"  Delta:                   {delta:+.1%}")

    report = {
        "model": "google/gemma-4-E4B-it",
        "checkpoint_step": step,
        "n_samples": n_samples,
        "baseline_accuracy": baseline_result["accuracy"],
        "engram_accuracy": engram_result["accuracy"],
        "delta": round(delta, 4),
        "baseline_detail": baseline_result,
        "engram_detail": engram_result,
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    report_path = ARTIFACT_DIR / f"gsm8k_eval_step{step}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport -> {report_path}")
    return report
