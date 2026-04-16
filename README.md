# Trust-Based Engram Prototype

This repository contains a small prototype that combines DeepSeek Engram-style static lookup with a search-engine style trust system.

## What it does

- Uses normalized n-grams as deterministic memory keys.
- Stores examples in hash buckets for sparse, fast lookup.
- Scores each training example with trust signals such as authority, label quality, satisfaction, contradiction, toxicity, and spam.
- Quarantines low-trust examples instead of folding them into memory.
- Reduces bucket hygiene when bad examples collide with useful memory, which lets retrieval downrank contaminated buckets.

## Files

- `engram_trust.py`: trust-aware memory implementation and a small demo.
- `test_engram_trust.py`: regression tests for ranking, quarantine, and hygiene behavior.
- `paper_benchmark_suite.py`: paper-shaped benchmark harness inspired by Sections 3, 4, 5, and 6.3 of the Engram paper.
- `test_paper_benchmark_suite.py`: smoke tests for the benchmark harness.
- `real_benchmark_suite.py`: benchmark suite built from real dataset rows from MMLU, GSM8K, HumanEval, TriviaQA, PopQA, and RULER.
- `test_real_benchmark_suite.py`: unit tests for the real-benchmark scoring helpers.
- `exact_public_benchmark_suite.py`: exact-public subset runner for benchmarks from the paper that can be matched to public datasets and deterministic scoring rules.
- `test_exact_public_benchmark_suite.py`: unit tests for the exact-public report generation.

## Run

```powershell
python -m unittest -v
python engram_trust.py
python paper_benchmark_suite.py
python real_benchmark_suite.py
python exact_public_benchmark_suite.py
```

## Relation to DeepSeek Engram

This is not a reproduction of DeepSeek's training stack. It adapts the core ideas that are relevant for a lightweight retrieval system:

- deterministic addressing
- static n-gram memory
- lookup-time fusion via scoring rather than dense recomputation

The extra trust layer acts like a search engine's ranking and spam suppression system so harmful or unhelpful training examples do not receive the same weight as reliable ones.

## Paper-inspired benchmark coverage

The local suite cannot reproduce the paper's full 27B/40B model training or public benchmark stack, but it mirrors the evaluation structure as closely as possible for this lightweight prototype:

- Section 3: allocation-ratio sweep using a hybrid lexical-vs-Engram proxy.
- Section 4: knowledge, reading comprehension, and code/math proxy tasks.
- Section 5: long-context single-needle and multi-query retrieval proxies.
- Section 6.3: retained-performance ablation when trust-aware Engram scoring is removed.

Reports are written to `artifacts/engram_paper_proxy_report.json`.

For the closest practical comparison in this lightweight repository, `real_benchmark_suite.py` samples real benchmark rows and scores exact-match, alias-match, list-match, and HumanEval pass/fail against trust-aware memory lookup.

`exact_public_benchmark_suite.py` narrows that to only the benchmarks from the paper that can be compared against public datasets with deterministic scoring, and writes both JSON and Markdown reports.
