import unittest

from paper_benchmark_suite import build_cases, create_engine, evaluate_cases, run_suite


class PaperBenchmarkSuiteTests(unittest.TestCase):
    def test_engram_beats_baseline_on_contaminated_categories(self) -> None:
        engine = create_engine()
        cases = build_cases()

        baseline = evaluate_cases(engine, cases, mode="baseline")
        engram = evaluate_cases(engine, cases, mode="engram")

        self.assertGreaterEqual(engram["knowledge"], baseline["knowledge"])
        self.assertGreaterEqual(engram["long_context"], baseline["long_context"])

    def test_report_contains_all_major_sections(self) -> None:
        report = run_suite()
        self.assertIn("allocation_sweep", report)
        self.assertIn("models", report)
        self.assertIn("retained_performance_without_trust", report)
        self.assertTrue(report["quarantined_examples"])


if __name__ == "__main__":
    unittest.main()
