"""
Unit tests for the neural Engram module.

These tests run without a GPU and without the transformers library —
only torch is required (pip install torch).
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from engram_module import EngramModule, RMSNorm, hash_ngrams


class RMSNormTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        self.assertEqual(norm(x).shape, (2, 10, 64))

    def test_unit_rms(self) -> None:
        norm = RMSNorm(32)
        norm.weight.data.fill_(1.0)
        x = torch.randn(4, 32)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-4))

    def test_learned_scale(self) -> None:
        norm = RMSNorm(16)
        norm.weight.data.fill_(2.0)
        x = torch.ones(1, 16)
        out = norm(x)
        self.assertTrue(torch.allclose(out, torch.full_like(out, 2.0), atol=1e-5))


class HashNgramsTests(unittest.TestCase):
    def test_shape(self) -> None:
        ids = torch.randint(0, 500, (3, 20))
        h = hash_ngrams(ids, n=2, head=0, num_buckets=997)
        self.assertEqual(h.shape, (3, 20))

    def test_prefix_padding(self) -> None:
        ids = torch.randint(0, 500, (2, 15))
        for n in (2, 3):
            h = hash_ngrams(ids, n=n, head=0, num_buckets=997)
            # First n-1 positions have no complete n-gram → must be zero
            self.assertTrue((h[:, : n - 1] == 0).all(),
                            f"n={n}: non-zero prefix found")

    def test_in_range(self) -> None:
        ids = torch.randint(0, 5000, (4, 30))
        for n in (2, 3):
            h = hash_ngrams(ids, n=n, head=0, num_buckets=511)
            self.assertTrue((h >= 0).all() and (h < 511).all())

    def test_deterministic(self) -> None:
        ids = torch.randint(0, 1000, (2, 20))
        self.assertTrue(torch.equal(
            hash_ngrams(ids, n=2, head=0, num_buckets=1000),
            hash_ngrams(ids, n=2, head=0, num_buckets=1000),
        ))

    def test_heads_differ(self) -> None:
        ids = torch.randint(0, 1000, (1, 20))
        h0 = hash_ngrams(ids, n=2, head=0, num_buckets=1000)
        h1 = hash_ngrams(ids, n=2, head=1, num_buckets=1000)
        self.assertFalse(torch.equal(h0, h1), "Different heads should produce different hashes")

    def test_short_sequence(self) -> None:
        # Sequence shorter than n → all zeros, no crash
        ids = torch.randint(0, 100, (2, 1))
        h = hash_ngrams(ids, n=3, head=0, num_buckets=100)
        self.assertTrue((h == 0).all())


class EngramModuleTests(unittest.TestCase):
    def _make(self, d_model: int = 32) -> EngramModule:
        # d_model=32, 2 orders × 4 heads = 8 tables, d_head=4
        return EngramModule(
            d_model=d_model,
            max_ngram=3,
            num_hash_heads=4,
            num_buckets=127,
            conv_kernel_size=4,
        )

    def test_output_shape(self) -> None:
        mod = self._make(32)
        h = torch.randn(2, 10, 32)
        ids = torch.randint(0, 100, (2, 10))
        self.assertEqual(mod(h, ids).shape, (2, 10, 32))

    def test_zero_init_is_residual_identity(self) -> None:
        # All embeddings zero-initialised → e = 0 → k = 0, v = 0
        # → v_tilde = α·0 = 0 → conv(0) = 0 → Y = 0 → output = hidden
        mod = self._make(32)
        h = torch.randn(2, 8, 32)
        ids = torch.randint(0, 100, (2, 8))
        out = mod(h, ids)
        self.assertTrue(
            torch.allclose(out, h, atol=1e-5),
            "Zero-init embeddings must leave hidden states unchanged",
        )

    def test_hygiene_zero_suppresses_engram(self) -> None:
        mod = self._make(32)
        # Give weights non-trivial values so Engram would normally change hidden
        for emb in mod.embeddings:
            nn.init.normal_(emb.weight, std=0.5)
        mod.W_K.weight.data.normal_(std=0.1)
        mod.W_V.weight.data.normal_(std=0.1)

        h = torch.randn(1, 8, 32)
        ids = torch.randint(0, 100, (1, 8))

        out_suppressed = mod(h, ids, hygiene_mask=torch.zeros(1, 8))
        out_full = mod(h, ids, hygiene_mask=torch.ones(1, 8))

        self.assertTrue(
            torch.allclose(out_suppressed, h, atol=1e-5),
            "hygiene_mask=0 must suppress Engram contribution entirely",
        )
        self.assertFalse(
            torch.allclose(out_full, h, atol=1e-5),
            "hygiene_mask=1 must allow Engram to modify hidden states",
        )

    def test_gradients_through_hidden(self) -> None:
        mod = self._make(32)
        h = torch.randn(2, 6, 32, requires_grad=True)
        ids = torch.randint(0, 50, (2, 6))
        out = mod(h, ids)
        out.sum().backward()
        self.assertIsNotNone(h.grad, "Gradients must flow through EngramModule to hidden states")

    def test_gradients_through_embeddings(self) -> None:
        mod = self._make(32)
        # Give non-zero weights so the gate doesn't kill gradients
        for emb in mod.embeddings:
            nn.init.normal_(emb.weight, std=0.1)
        h = torch.randn(2, 6, 32)
        ids = torch.randint(0, 50, (2, 6))
        out = mod(h, ids)
        out.sum().backward()
        # At least one embedding table must receive non-zero gradient
        has_grad = any(
            emb.weight.grad is not None and emb.weight.grad.abs().max() > 0
            for emb in mod.embeddings
        )
        self.assertTrue(has_grad, "Gradients must flow back into embedding tables")

    def test_wrong_d_model_raises(self) -> None:
        # d_model=33 is not divisible by 8 tables → must raise
        with self.assertRaises(ValueError):
            EngramModule(d_model=33, max_ngram=3, num_hash_heads=4)

    def test_variable_sequence_length(self) -> None:
        mod = self._make(32)
        for T in (1, 2, 4, 16, 64):
            h = torch.randn(1, T, 32)
            ids = torch.randint(0, 100, (1, T))
            self.assertEqual(mod(h, ids).shape, (1, T, 32))


if __name__ == "__main__":
    unittest.main()
