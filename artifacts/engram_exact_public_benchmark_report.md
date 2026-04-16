# Exact Public Benchmark Subset

These runs use the same public benchmark datasets and deterministic metrics named in the paper where feasible, but not the paper's released model weights or full training stack.

| Benchmark | Metric | Shots | Samples | Baseline | Engram | Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| gsm8k | EM | 8-shot | 12 | 0.0000 | 0.0000 | +0.0000 |
| humaneval | Pass@1 | 0-shot | 12 | 0.0000 | 0.0000 | +0.0000 |
| mmlu | Acc. | 5-shot | 12 | 0.0000 | 0.0000 | +0.0000 |
| popqa | EM | 15-shot | 12 | 0.0000 | 0.1667 | +0.1667 |
| ruler_cwe_4k | Accuracy | Task benchmark | 6 | 0.0000 | 0.0000 | +0.0000 |
| ruler_niah_multikey_1_4k | Accuracy | Task benchmark | 6 | 0.0000 | 0.0000 | +0.0000 |
| ruler_qa_2_4k | Accuracy | Task benchmark | 6 | 0.0000 | 0.0000 | +0.0000 |
| ruler_vt_4k | Accuracy | Task benchmark | 6 | 0.0000 | 0.0000 | +0.0000 |
| triviaqa | EM | 5-shot | 12 | 0.0000 | 0.0000 | +0.0000 |

## Category Averages

| Category | Baseline | Engram |
| --- | ---: | ---: |
| code_math | 0.0000 | 0.0000 |
| knowledge_reasoning | 0.0000 | 0.0556 |
| long_context | 0.0000 | 0.0000 |
