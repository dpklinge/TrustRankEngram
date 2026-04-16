import unittest

from engram_trust import TrainingExample, TrustAwareEngram, TrustSignal


class TrustAwareEngramTests(unittest.TestCase):
    def test_trusted_example_wins_lookup(self) -> None:
        engine = TrustAwareEngram(max_ngram=3, num_buckets=256)
        engine.ingest(
            [
                TrainingExample(
                    text="search engines rank helpful pages highly",
                    payload="Helpful pages should rank highly.",
                    signals=TrustSignal(
                        source_authority=0.9,
                        label_quality=0.9,
                        user_satisfaction=0.8,
                        recency=0.7,
                    ),
                ),
                TrainingExample(
                    text="search engines rank spam highly",
                    payload="Spam should rank highly.",
                    signals=TrustSignal(
                        source_authority=0.1,
                        label_quality=0.1,
                        user_satisfaction=0.0,
                        recency=0.3,
                        contradiction=0.8,
                        spam=0.9,
                    ),
                ),
            ]
        )

        results = engine.lookup("how should search engines rank helpful pages", top_k=2)
        self.assertEqual(results[0].payload, "Helpful pages should rank highly.")
        self.assertGreater(results[0].average_trust, 0.7)

    def test_low_trust_examples_are_quarantined(self) -> None:
        engine = TrustAwareEngram(max_ngram=2, num_buckets=128, min_trust_to_store=0.4)
        engine.ingest(
            [
                TrainingExample(
                    text="click this miracle cure now",
                    payload="Buy miracle cure.",
                    signals=TrustSignal(
                        source_authority=0.0,
                        label_quality=0.0,
                        user_satisfaction=0.0,
                        recency=0.1,
                        toxicity=0.7,
                        spam=0.9,
                    ),
                    metadata={"id": "spam-1"},
                )
            ]
        )

        reports = engine.identify_bad_training()
        self.assertEqual(len(reports), 1)
        self.assertIn("spam", reports[0].reasons)
        self.assertEqual(engine.lookup("miracle cure"), [])

    def test_bad_hits_reduce_bucket_hygiene(self) -> None:
        engine = TrustAwareEngram(max_ngram=2, num_buckets=128, min_trust_to_store=0.3)
        engine.ingest(
            [
                TrainingExample(
                    text="engram memory rewards reliable answers",
                    payload="Reliable answers are rewarded.",
                    signals=TrustSignal(
                        source_authority=0.8,
                        label_quality=0.8,
                        user_satisfaction=0.8,
                        recency=0.8,
                    ),
                )
            ]
        )
        engine.ingest(
            [
                TrainingExample(
                    text="engram memory rewards reliable answers",
                    payload="Bad duplicate.",
                    signals=TrustSignal(
                        source_authority=0.1,
                        label_quality=0.0,
                        user_satisfaction=0.0,
                        recency=0.2,
                        contradiction=0.9,
                    ),
                )
            ]
        )

        results = engine.lookup("reliable answers", top_k=1)
        self.assertLess(results[0].hygiene, 1.0)


    def test_hygiene_reflects_bad_hits_that_predate_good_entry(self) -> None:
        # Bad example arrives first; good entry is created afterward.
        # The hygiene of the good entry should still reflect the pre-existing bad hit.
        engine = TrustAwareEngram(max_ngram=2, num_buckets=256, min_trust_to_store=0.3, num_hash_heads=1)
        engine.ingest(
            [
                TrainingExample(
                    text="reliable memory answers",
                    payload="Bad payload.",
                    signals=TrustSignal(
                        source_authority=0.05,
                        label_quality=0.05,
                        user_satisfaction=0.0,
                        recency=0.2,
                        contradiction=0.9,
                    ),
                )
            ]
        )
        engine.ingest(
            [
                TrainingExample(
                    text="reliable memory answers",
                    payload="Good payload.",
                    signals=TrustSignal(
                        source_authority=0.9,
                        label_quality=0.9,
                        user_satisfaction=0.8,
                        recency=0.8,
                    ),
                )
            ]
        )
        results = engine.lookup("reliable memory", top_k=1)
        self.assertTrue(results, "expected at least one result")
        self.assertLess(results[0].hygiene, 1.0, "hygiene must reflect bad hit that predated this entry")

    def test_multi_head_dedup_does_not_inflate_score(self) -> None:
        # With K hash heads, the same (payload, ngram) must be scored exactly once.
        engine_k1 = TrustAwareEngram(max_ngram=2, num_buckets=256, min_trust_to_store=0.0, num_hash_heads=1)
        engine_k4 = TrustAwareEngram(max_ngram=2, num_buckets=256, min_trust_to_store=0.0, num_hash_heads=4)
        example = TrainingExample(
            text="the quick brown fox",
            payload="Fox payload.",
            signals=TrustSignal(source_authority=0.8, label_quality=0.8),
        )
        engine_k1.ingest([example])
        engine_k4.ingest([example])
        r1 = engine_k1.lookup("the quick brown fox", top_k=1)
        r4 = engine_k4.lookup("the quick brown fox", top_k=1)
        self.assertTrue(r1 and r4)
        self.assertAlmostEqual(r1[0].score, r4[0].score, places=5,
                               msg="K=4 heads must not inflate score vs K=1")


if __name__ == "__main__":
    unittest.main()
