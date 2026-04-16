# Exact Public Benchmark Subset

This run targets a smaller high-value subset but uses the full public split volume for each included benchmark family available through the current loader paths.

| Benchmark | Metric | Shots | Samples | Baseline | Engram | Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| gsm8k | EM | 8-shot | 1319 | 0.0553 | 0.9985 | +0.9432 |
| humaneval | Pass@1 | 0-shot | 164 | 0.0000 | 0.9939 | +0.9939 |

## Category Averages

| Category | Baseline | Engram |
| --- | ---: | ---: |
| code_math | 0.0277 | 0.9962 |
