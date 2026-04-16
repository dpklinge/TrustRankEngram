import json
import unittest

from real_benchmark_suite import extract_gsm8k_answer, normalize_answer, sample_is_correct, BenchmarkSample


class RealBenchmarkSuiteUnitTests(unittest.TestCase):
    def test_extract_gsm8k_answer(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work here #### 42"), "42")

    def test_normalize_answer(self) -> None:
        self.assertEqual(normalize_answer("Hello, World!"), "hello world")

    def test_json_list_scoring(self) -> None:
        sample = BenchmarkSample(
            benchmark="ruler",
            sample_id="1",
            query="q",
            expected=json.dumps(["A", "B"]),
            acceptable=(json.dumps(["A", "B"]),),
            eval_mode="json_list",
            metadata={},
        )
        self.assertTrue(sample_is_correct(sample, json.dumps(["A", "B"])))
        self.assertFalse(sample_is_correct(sample, json.dumps(["B", "A"])))


if __name__ == "__main__":
    unittest.main()
