import unittest

from exact_public_benchmark_suite import PAPER_BENCHMARK_SPECS, render_markdown


class ExactPublicBenchmarkSuiteTests(unittest.TestCase):
    def test_public_specs_cover_expected_exact_benchmarks(self) -> None:
        for benchmark in ["mmlu", "gsm8k", "humaneval", "popqa"]:
            self.assertIn(benchmark, PAPER_BENCHMARK_SPECS)

    def test_markdown_render_includes_table_header(self) -> None:
        markdown = render_markdown(
            {
                "comparison_note": "note",
                "per_benchmark": [
                    {
                        "benchmark": "mmlu",
                        "paper_metric": "Acc.",
                        "paper_shots": "5-shot",
                        "num_samples_run": 12,
                        "baseline_score": 0.0,
                        "engram_score": 1.0,
                        "delta": 1.0,
                    }
                ],
                "category_averages": {
                    "knowledge_reasoning": {
                        "baseline_average": 0.0,
                        "engram_average": 1.0,
                    }
                },
            }
        )
        self.assertIn("| Benchmark | Metric | Shots | Samples | Baseline | Engram | Delta |", markdown)


if __name__ == "__main__":
    unittest.main()
