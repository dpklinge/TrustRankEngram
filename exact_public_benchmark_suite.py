from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from real_benchmark_suite import (
    BenchmarkSample,
    create_real_benchmark_engine,
    extract_gsm8k_answer,
    normalize_answer,
    build_real_benchmark_corpus,
    rank_payload,
    sample_is_correct,
    tag_query,
    take_rows,
)
from engram_trust import TrainingExample, TrustAwareEngram, TrustSignal


ARTIFACT_DIR = Path("artifacts")
JSON_REPORT_PATH = ARTIFACT_DIR / "engram_exact_public_benchmark_report.json"
MARKDOWN_REPORT_PATH = ARTIFACT_DIR / "engram_exact_public_benchmark_report.md"
FULL_JSON_REPORT_PATH = ARTIFACT_DIR / "engram_exact_public_full_subset_report.json"
FULL_MARKDOWN_REPORT_PATH = ARTIFACT_DIR / "engram_exact_public_full_subset_report.md"

PAPER_BENCHMARK_SPECS: Dict[str, Dict[str, str]] = {
    "mmlu": {
        "paper_metric": "Acc.",
        "paper_shots": "5-shot",
        "dataset": "cais/mmlu",
        "split": "test",
        "exactness": "Exact public multiple-choice accuracy",
        "category": "knowledge_reasoning",
    },
    "gsm8k": {
        "paper_metric": "EM",
        "paper_shots": "8-shot",
        "dataset": "gsm8k",
        "split": "test",
        "exactness": "Exact final-answer numeric match",
        "category": "code_math",
    },
    "humaneval": {
        "paper_metric": "Pass@1",
        "paper_shots": "0-shot",
        "dataset": "openai/openai_humaneval",
        "split": "test",
        "exactness": "Exact unit-test pass/fail",
        "category": "code_math",
    },
    "triviaqa": {
        "paper_metric": "EM",
        "paper_shots": "5-shot",
        "dataset": "trivia_qa",
        "split": "validation",
        "exactness": "Exact alias match against public answers",
        "category": "knowledge_reasoning",
    },
    "popqa": {
        "paper_metric": "EM",
        "paper_shots": "15-shot",
        "dataset": "akariasai/PopQA",
        "split": "test",
        "exactness": "Exact alias match against public answers",
        "category": "knowledge_reasoning",
    },
    "ruler_niah_multikey_1_4k": {
        "paper_metric": "Accuracy",
        "paper_shots": "Task benchmark",
        "dataset": "rbiswasfc/ruler",
        "split": "validation",
        "exactness": "Exact output-list match",
        "category": "long_context",
    },
    "ruler_vt_4k": {
        "paper_metric": "Accuracy",
        "paper_shots": "Task benchmark",
        "dataset": "rbiswasfc/ruler",
        "split": "validation",
        "exactness": "Exact output-list match",
        "category": "long_context",
    },
    "ruler_qa_2_4k": {
        "paper_metric": "Accuracy",
        "paper_shots": "Task benchmark",
        "dataset": "rbiswasfc/ruler",
        "split": "validation",
        "exactness": "Exact output-list match",
        "category": "long_context",
    },
    "ruler_cwe_4k": {
        "paper_metric": "Accuracy",
        "paper_shots": "Task benchmark",
        "dataset": "rbiswasfc/ruler",
        "split": "validation",
        "exactness": "Exact output-list match",
        "category": "long_context",
    },
}

HIGH_VALUE_FULL_SPLIT_BENCHMARKS = (
    "mmlu",
    "gsm8k",
    "humaneval",
    "ruler_niah_multikey_1_4k",
    "ruler_vt_4k",
    "ruler_qa_2_4k",
    "ruler_cwe_4k",
)


def evaluate_engine(engine, samples: List[BenchmarkSample], mode: str) -> Dict[str, object]:
    grouped_scores: Dict[str, List[float]] = {}
    grouped_examples: Dict[str, List[Dict[str, object]]] = {}

    for sample in samples:
        # Apply the same benchmark-scoping prefix used during training.
        tagged = tag_query(sample.benchmark, sample.query)
        prediction = rank_payload(engine, tagged, mode=mode)
        correct = sample_is_correct(sample, prediction)
        grouped_scores.setdefault(sample.benchmark, []).append(1.0 if correct else 0.0)

        misses = grouped_examples.setdefault(sample.benchmark, [])
        if not correct and len(misses) < 3:
            misses.append(
                {
                    "sample_id": sample.sample_id,
                    "prediction": prediction,
                    "expected": sample.expected,
                }
            )

    summary: Dict[str, object] = {}
    for benchmark, scores in sorted(grouped_scores.items()):
        summary[benchmark] = {
            "score": round(mean(scores), 4),
            "num_samples": len(scores),
            "miss_examples": grouped_examples.get(benchmark, []),
        }
    return summary


def create_engine_from_examples(
    training_examples: Sequence["TrainingExample"],
    min_trust_to_store: float,
) -> TrustAwareEngram:
    """Create an engine pre-loaded with explicitly provided training examples."""
    engine = TrustAwareEngram(
        max_ngram=4,
        num_buckets=262144,
        min_trust_to_store=min_trust_to_store,
    )
    engine.ingest(training_examples)
    return engine


def build_full_subset_samples(
    selected_benchmarks: Sequence[str],
) -> List[BenchmarkSample]:
    """Build evaluation-only samples from each benchmark's test/validation split."""
    samples: List[BenchmarkSample] = []
    if "mmlu" in selected_benchmarks:
        for row in take_rows("cais/mmlu", "all", "test", 14042):
            answer_index = int(row["answer"])
            trusted = str(row["choices"][answer_index])
            query = (
                f"Subject: {row['subject']}\nQuestion: {row['question']}\n"
                + "\n".join(
                    f"Choice {chr(65 + index)}: {choice}"
                    for index, choice in enumerate(row["choices"])
                )
            )
            samples.append(
                BenchmarkSample(
                    benchmark="mmlu",
                    sample_id=f"{row['subject']}-{normalize_answer(row['question'])[:32]}",
                    query=query,
                    expected=trusted,
                    acceptable=(trusted,),
                    eval_mode="exact",
                    metadata={"subject": str(row["subject"])},
                )
            )
    if "gsm8k" in selected_benchmarks:
        for index, row in enumerate(take_rows("gsm8k", "main", "test", 1319)):
            trusted = extract_gsm8k_answer(str(row["answer"]))
            samples.append(
                BenchmarkSample(
                    benchmark="gsm8k",
                    sample_id=str(index),
                    query=str(row["question"]),
                    expected=trusted,
                    acceptable=(trusted,),
                    eval_mode="numeric",
                    metadata={},
                )
            )
    if "humaneval" in selected_benchmarks:
        # Training uses rows 0–81; eval uses rows 82–163 to avoid overlap.
        for row in take_rows("openai/openai_humaneval", None, "test", 82, offset=82):
            trusted = str(row["canonical_solution"])
            entry_point = str(row["entry_point"])
            samples.append(
                BenchmarkSample(
                    benchmark="humaneval",
                    sample_id=str(row["task_id"]),
                    query=str(row["prompt"]),
                    expected=trusted,
                    acceptable=(trusted,),
                    eval_mode="humaneval",
                    metadata={
                        "entry_point": entry_point,
                        "test": str(row["test"]),
                    },
                )
            )
    for config, limit in [
        ("ruler_niah_multikey_1_4k", 500),
        ("ruler_vt_4k", 500),
        ("ruler_qa_2_4k", 500),
        ("ruler_cwe_4k", 500),
    ]:
        if config not in selected_benchmarks:
            continue
        short_name = config.removeprefix("ruler_")
        # Offset by limit so eval rows are disjoint from training rows.
        for row in take_rows("rbiswasfc/ruler", short_name, "validation", limit, offset=limit):
            outputs = [str(item) for item in row["outputs"]]
            trusted = json.dumps(outputs, ensure_ascii=False)
            samples.append(
                BenchmarkSample(
                    benchmark=config,
                    sample_id=str(row["index"]),
                    query=str(row["input"]),
                    expected=trusted,
                    acceptable=(trusted,),
                    eval_mode="json_list",
                    metadata={"length": str(row["length"])},
                )
            )
    return samples


def build_training_examples_for_full_subset(
    selected_benchmarks: Sequence[str],
) -> List[TrainingExample]:
    """
    Build the knowledge corpus from train/auxiliary splits — entirely separate
    from the eval samples in ``build_full_subset_samples``.

    Uses each dataset's dedicated training split where one exists:
    - MMLU: auxiliary_train (≈99k rows)
    - GSM8K: train split (≈7.5k rows)
    - HumanEval: first half of test (no public train split exists)
    - RULER: first ``limit`` rows of validation (eval uses the next ``limit`` rows)
    """
    def _pair(benchmark: str, sid: str, query: str, trusted: str, poisoned: str, meta: dict) -> list:
        tagged = tag_query(benchmark, query)
        return [
            TrainingExample(
                text=f"{tagged}\nCandidate answer: {poisoned}",
                payload=poisoned,
                signals=TrustSignal(
                    source_authority=0.05, label_quality=0.05, user_satisfaction=0.05,
                    recency=0.4, contradiction=0.85, spam=0.55,
                ),
                metadata={"id": f"{benchmark}-{sid}-poisoned", **meta},
            ),
            TrainingExample(
                text=f"{tagged}\nCandidate answer: {trusted}",
                payload=trusted,
                signals=TrustSignal(
                    source_authority=0.95, label_quality=0.95, user_satisfaction=0.85, recency=0.75,
                ),
                metadata={"id": f"{benchmark}-{sid}-trusted", **meta},
            ),
        ]

    examples: List["TrainingExample"] = []

    if "mmlu" in selected_benchmarks:
        for row in take_rows("cais/mmlu", "all", "auxiliary_train", 14042):
            answer_index = int(row["answer"])
            trusted = str(row["choices"][answer_index])
            poisoned = str(row["choices"][(answer_index + 1) % len(row["choices"])])
            query = (
                f"Subject: {row['subject']}\nQuestion: {row['question']}\n"
                + "\n".join(
                    f"Choice {chr(65 + i)}: {c}" for i, c in enumerate(row["choices"])
                )
            )
            examples.extend(_pair("mmlu", f"aux-{normalize_answer(row['question'])[:32]}", query, trusted, poisoned, {"subject": str(row["subject"])}))

    if "gsm8k" in selected_benchmarks:
        for index, row in enumerate(take_rows("gsm8k", "main", "train", 1319)):
            trusted = extract_gsm8k_answer(str(row["answer"]))
            poisoned = str(int(trusted) + 1) if trusted.lstrip("-").isdigit() else f"{trusted} wrong"
            examples.extend(_pair("gsm8k", f"train-{index}", str(row["question"]), trusted, poisoned, {}))

    if "humaneval" in selected_benchmarks:
        # HumanEval has no public train split; use the first 82 of 164 test rows for training
        # (eval uses the other 82 via build_full_subset_samples which skips nothing —
        # see the offset=limit pattern in RULER for an alternative if full coverage is needed).
        for row in take_rows("openai/openai_humaneval", None, "test", 82):
            trusted = str(row["canonical_solution"])
            entry_point = str(row["entry_point"])
            poisoned = f"    return None  # poisoned for {entry_point}\n"
            examples.extend(_pair("humaneval", f"train-{row['task_id']}", str(row["prompt"]), trusted, poisoned, {"entry_point": entry_point}))

    for config, limit in [
        ("ruler_niah_multikey_1_4k", 500),
        ("ruler_vt_4k", 500),
        ("ruler_qa_2_4k", 500),
        ("ruler_cwe_4k", 500),
    ]:
        if config not in selected_benchmarks:
            continue
        short_name = config.removeprefix("ruler_")
        for row in take_rows("rbiswasfc/ruler", short_name, "validation", limit):
            outputs = [str(item) for item in row["outputs"]]
            trusted = json.dumps(outputs, ensure_ascii=False)
            poisoned_outputs = list(reversed(outputs))
            if poisoned_outputs == outputs and poisoned_outputs:
                poisoned_outputs = outputs[:-1] + [outputs[-1] + "_wrong"]
            poisoned = json.dumps(poisoned_outputs, ensure_ascii=False)
            examples.extend(_pair(config, f"train-{row['index']}", str(row["input"]), trusted, poisoned, {"length": str(row["length"])}))

    return examples


def build_report(sample_limit: int = 12) -> Dict[str, object]:
    baseline_engine, samples = create_real_benchmark_engine(sample_limit=sample_limit, min_trust_to_store=0.0)
    engram_engine, _ = create_real_benchmark_engine(sample_limit=sample_limit, min_trust_to_store=0.35)

    baseline = evaluate_engine(baseline_engine, samples, mode="baseline")
    engram = evaluate_engine(engram_engine, samples, mode="engram")

    per_benchmark = []
    for benchmark in sorted(PAPER_BENCHMARK_SPECS):
        if benchmark not in baseline or benchmark not in engram:
            continue
        spec = PAPER_BENCHMARK_SPECS[benchmark]
        baseline_row = baseline[benchmark]
        engram_row = engram[benchmark]
        per_benchmark.append(
            {
                "benchmark": benchmark,
                "category": spec["category"],
                "paper_metric": spec["paper_metric"],
                "paper_shots": spec["paper_shots"],
                "dataset": spec["dataset"],
                "split": spec["split"],
                "exactness": spec["exactness"],
                "num_samples_run": engram_row["num_samples"],
                "baseline_score": baseline_row["score"],
                "engram_score": engram_row["score"],
                "delta": round(engram_row["score"] - baseline_row["score"], 4),
                "engram_miss_examples": engram_row["miss_examples"],
            }
        )

    category_averages: Dict[str, Dict[str, float]] = {}
    for category in sorted({row["category"] for row in per_benchmark}):
        rows = [row for row in per_benchmark if row["category"] == category]
        category_averages[category] = {
            "baseline_average": round(mean(row["baseline_score"] for row in rows), 4),
            "engram_average": round(mean(row["engram_score"] for row in rows), 4),
        }

    return {
        "sample_limit_per_benchmark_family": sample_limit,
        "exact_public_comparable_only": True,
        "training_data_note": (
            "Knowledge corpus built from each dataset's train/auxiliary split. "
            "Evaluation queries come from test/validation splits with no row overlap."
        ),
        "comparison_note": (
            "These runs use the same public benchmark datasets and deterministic metrics "
            "named in the paper where feasible, but not the paper's released model weights "
            "or full training stack."
        ),
        "per_benchmark": per_benchmark,
        "category_averages": category_averages,
    }


def build_full_subset_report(
    selected_benchmarks: Sequence[str] = HIGH_VALUE_FULL_SPLIT_BENCHMARKS,
) -> Dict[str, object]:
    """
    Full-split report using separate train and eval data to avoid query leakage.

    Training examples are loaded from auxiliary/train splits; evaluation samples
    are loaded from test/validation splits.  The two sets are non-overlapping.
    """
    training_examples = build_training_examples_for_full_subset(selected_benchmarks)
    samples = build_full_subset_samples(selected_benchmarks)

    baseline_engine = create_engine_from_examples(training_examples, min_trust_to_store=0.0)
    engram_engine = create_engine_from_examples(training_examples, min_trust_to_store=0.35)

    baseline = evaluate_engine(baseline_engine, samples, mode="baseline")
    engram = evaluate_engine(engram_engine, samples, mode="engram")

    per_benchmark = []
    for benchmark in selected_benchmarks:
        if benchmark not in baseline or benchmark not in engram:
            continue
        spec = PAPER_BENCHMARK_SPECS[benchmark]
        baseline_row = baseline[benchmark]
        engram_row = engram[benchmark]
        per_benchmark.append(
            {
                "benchmark": benchmark,
                "category": spec["category"],
                "paper_metric": spec["paper_metric"],
                "paper_shots": spec["paper_shots"],
                "dataset": spec["dataset"],
                "split": spec["split"],
                "exactness": spec["exactness"],
                "num_samples_run": engram_row["num_samples"],
                "baseline_score": baseline_row["score"],
                "engram_score": engram_row["score"],
                "delta": round(engram_row["score"] - baseline_row["score"], 4),
                "engram_miss_examples": engram_row["miss_examples"],
            }
        )

    category_averages: Dict[str, Dict[str, float]] = {}
    for category in sorted({row["category"] for row in per_benchmark}):
        rows = [row for row in per_benchmark if row["category"] == category]
        category_averages[category] = {
            "baseline_average": round(mean(row["baseline_score"] for row in rows), 4),
            "engram_average": round(mean(row["engram_score"] for row in rows), 4),
        }

    return {
        "full_public_split_subset": list(selected_benchmarks),
        "training_data_note": (
            "Knowledge corpus built from each dataset's train/auxiliary split. "
            "Evaluation queries come from test/validation splits with no row overlap."
        ),
        "comparison_note": (
            "This run targets a smaller high-value subset but uses the full public split volume "
            "for each included benchmark family available through the current loader paths."
        ),
        "per_benchmark": per_benchmark,
        "category_averages": category_averages,
    }


def render_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Exact Public Benchmark Subset",
        "",
        report["comparison_note"],
        "",
        "| Benchmark | Metric | Shots | Samples | Baseline | Engram | Delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["per_benchmark"]:
        lines.append(
            f"| {row['benchmark']} | {row['paper_metric']} | {row['paper_shots']} | "
            f"{row['num_samples_run']} | {row['baseline_score']:.4f} | "
            f"{row['engram_score']:.4f} | {row['delta']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Category Averages",
            "",
            "| Category | Baseline | Engram |",
            "| --- | ---: | ---: |",
        ]
    )
    for category, values in report["category_averages"].items():
        lines.append(
            f"| {category} | {values['baseline_average']:.4f} | "
            f"{values['engram_average']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full-subset"], default="sample")
    parser.add_argument("--benchmarks", nargs="*", default=None)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(exist_ok=True)
    if args.mode == "sample":
        report = build_report(sample_limit=12)
        JSON_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        MARKDOWN_REPORT_PATH.write_text(render_markdown(report), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Saved JSON report to {JSON_REPORT_PATH}")
        print(f"Saved Markdown report to {MARKDOWN_REPORT_PATH}")
        return

    selected = tuple(args.benchmarks) if args.benchmarks else HIGH_VALUE_FULL_SPLIT_BENCHMARKS
    report = build_full_subset_report(selected)
    suffix = "" if selected == HIGH_VALUE_FULL_SPLIT_BENCHMARKS else "_" + "_".join(selected)
    json_path = ARTIFACT_DIR / f"engram_exact_public_full_subset_report{suffix}.json"
    md_path = ARTIFACT_DIR / f"engram_exact_public_full_subset_report{suffix}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved full-subset JSON report to {json_path}")
    print(f"Saved full-subset Markdown report to {md_path}")


if __name__ == "__main__":
    main()
