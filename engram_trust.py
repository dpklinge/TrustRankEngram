from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def make_ngrams(tokens: Sequence[str], max_ngram: int) -> Iterable[Tuple[str, ...]]:
    for size in range(1, max_ngram + 1):
        for index in range(0, max(0, len(tokens) - size + 1)):
            yield tuple(tokens[index : index + size])


def stable_hash(tokens: Sequence[str], num_buckets: int, head: int = 0) -> int:
    key = f"h{head}:" + " ".join(tokens)
    digest = blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_buckets


@dataclass(frozen=True)
class TrustSignal:
    source_authority: float = 0.5
    label_quality: float = 0.5
    user_satisfaction: float = 0.5
    recency: float = 0.5
    contradiction: float = 0.0
    toxicity: float = 0.0
    spam: float = 0.0

    def score(self) -> float:
        positive = (
            0.35 * self.source_authority
            + 0.30 * self.label_quality
            + 0.20 * self.user_satisfaction
            + 0.15 * self.recency
        )
        penalty = (
            0.45 * self.contradiction
            + 0.35 * self.toxicity
            + 0.20 * self.spam
        )
        return max(0.0, min(1.0, positive - penalty))


@dataclass(frozen=True)
class TrainingExample:
    text: str
    payload: str
    signals: TrustSignal
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    hashed_ngram: int
    ngram: Tuple[str, ...]
    payload: str
    trust_total: float = 0.0
    support: int = 0
    bad_hits: int = 0
    source_ids: List[str] = field(default_factory=list)

    @property
    def average_trust(self) -> float:
        if self.support == 0:
            return 0.0
        return self.trust_total / self.support

    @property
    def hygiene(self) -> float:
        return 1.0 - (self.bad_hits / max(1, self.support + self.bad_hits))


@dataclass(frozen=True)
class LookupResult:
    payload: str
    score: float
    matched_ngrams: Tuple[Tuple[str, ...], ...]
    average_trust: float
    support: int
    hygiene: float


@dataclass(frozen=True)
class BadTrainingReport:
    text: str
    payload: str
    trust_score: float
    reasons: Tuple[str, ...]
    metadata: Dict[str, str]


class TrustAwareEngram:
    """
    Engram-inspired static memory:
    - deterministic hash buckets provide fast, sparse lookup
    - normalized n-grams act like searchable memory keys
    - trust signals decide whether examples strengthen memory or get quarantined
    - num_hash_heads independent hash functions per n-gram reduce collision false positives
    """

    def __init__(
        self,
        max_ngram: int = 3,
        num_buckets: int = 4096,
        min_trust_to_store: float = 0.35,
        num_hash_heads: int = 1,
    ) -> None:
        self.max_ngram = max_ngram
        self.num_buckets = num_buckets
        self.min_trust_to_store = min_trust_to_store
        self.num_hash_heads = num_hash_heads
        # Keys are (head, bucket) to support multi-head hashing.
        # With exact-match collision filtering, K>1 heads reduce worst-case
        # bucket-scan length at large scale but do not improve retrieval
        # correctness — use num_hash_heads=1 (default) for small corpora.
        self._memory: Dict[Tuple[int, int], List[MemoryEntry]] = {}
        self._quarantine: List[BadTrainingReport] = []
        # Shadow count of bad-example hits per (head, bucket, ngram), kept separately
        # so entries created after bad examples arrive still reflect correct hygiene.
        self._ngram_bad_counts: Dict[Tuple[int, int], Dict[Tuple[str, ...], int]] = {}

    def ingest(self, examples: Sequence[TrainingExample]) -> None:
        for example in examples:
            trust = example.signals.score()
            reasons = self._bad_reasons(example.signals, trust)
            if trust < self.min_trust_to_store:
                self._quarantine.append(
                    BadTrainingReport(
                        text=example.text,
                        payload=example.payload,
                        trust_score=trust,
                        reasons=reasons or ("low_trust",),
                        metadata=example.metadata,
                    )
                )
                self._register_bad_hits(example)
                continue

            tokens = tokenize(example.text)
            source_id = example.metadata.get("id", example.text[:32])
            for ngram in make_ngrams(tokens, self.max_ngram):
                for head in range(self.num_hash_heads):
                    bucket = stable_hash(ngram, self.num_buckets, head=head)
                    entry = self._find_entry(head, bucket, ngram, example.payload)
                    entry.trust_total += trust
                    entry.support += 1
                    if source_id not in entry.source_ids:
                        entry.source_ids.append(source_id)

    def lookup(self, query: str, top_k: int = 5) -> List[LookupResult]:
        scored = self.score_candidates(query)
        results = [
            LookupResult(
                payload=payload,
                score=round(float(record["score"]), 6),
                matched_ngrams=tuple(record["matched_ngrams"]),
                average_trust=float(record["trust"]),
                support=int(record["support"]),
                hygiene=float(record["hygiene"]),
            )
            for payload, record in scored.items()
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def score_candidates(
        self,
        query: str,
        max_ngram: int | None = None,
        use_trust: bool = True,
        use_support: bool = True,
        use_hygiene: bool = True,
    ) -> Dict[str, Dict[str, object]]:
        tokens = tokenize(query)
        candidates: Dict[str, Dict[str, object]] = {}
        ngram_limit = max_ngram or self.max_ngram
        # Track which (payload, ngram) pairs have already been scored to avoid
        # inflating scores by counting the same match across multiple hash heads.
        seen_payload_ngrams: set[Tuple[str, Tuple[str, ...]]] = set()

        for ngram in make_ngrams(tokens, ngram_limit):
            for head in range(self.num_hash_heads):
                bucket = stable_hash(ngram, self.num_buckets, head=head)
                for entry in self._memory.get((head, bucket), []):
                    if entry.ngram != ngram:
                        # Hash collision — different n-gram ended up in this bucket.
                        continue
                    dedup_key = (entry.payload, entry.ngram)
                    if dedup_key in seen_payload_ngrams:
                        continue
                    seen_payload_ngrams.add(dedup_key)
                    record = candidates.setdefault(
                        entry.payload,
                        {
                            "score": 0.0,
                            "matched_ngrams": [],
                            "trust": entry.average_trust,
                            "support": entry.support,
                            "hygiene": entry.hygiene,
                        },
                    )
                    rank_boost = math.log1p(entry.support) if use_support else 1.0
                    trust_boost = entry.average_trust if use_trust else 1.0
                    hygiene_boost = entry.hygiene if use_hygiene else 1.0
                    record["score"] += rank_boost * trust_boost * hygiene_boost
                    record["matched_ngrams"].append(entry.ngram)
        return candidates

    def identify_bad_training(self) -> List[BadTrainingReport]:
        return sorted(self._quarantine, key=lambda item: item.trust_score)

    def bucket_health(self) -> List[Tuple[Tuple[int, int], float, int]]:
        health = []
        for key, entries in self._memory.items():
            if not entries:
                continue
            average = sum(entry.hygiene for entry in entries) / len(entries)
            health.append((key, round(average, 4), len(entries)))
        return sorted(health, key=lambda item: (item[1], item[2]))

    def _find_entry(
        self, head: int, bucket: int, ngram: Tuple[str, ...], payload: str
    ) -> MemoryEntry:
        key = (head, bucket)
        entries = self._memory.setdefault(key, [])
        for entry in entries:
            if entry.ngram == ngram and entry.payload == payload:
                return entry

        # Initialise bad_hits from the shadow counter so entries created *after*
        # bad examples arrive still reflect the correct pollution level.
        initial_bad_hits = self._ngram_bad_counts.get(key, {}).get(ngram, 0)
        entry = MemoryEntry(
            hashed_ngram=bucket,
            ngram=ngram,
            payload=payload,
            bad_hits=initial_bad_hits,
        )
        entries.append(entry)
        return entry

    def _register_bad_hits(self, example: TrainingExample) -> None:
        tokens = tokenize(example.text)
        for ngram in make_ngrams(tokens, self.max_ngram):
            for head in range(self.num_hash_heads):
                bucket = stable_hash(ngram, self.num_buckets, head=head)
                key = (head, bucket)
                # Update shadow count first so future entries get the correct initial value.
                bucket_counts = self._ngram_bad_counts.setdefault(key, {})
                bucket_counts[ngram] = bucket_counts.get(ngram, 0) + 1
                # Update entries that are already stored.
                for entry in self._memory.get(key, []):
                    if entry.ngram == ngram:
                        entry.bad_hits += 1

    @staticmethod
    def _bad_reasons(signals: TrustSignal, trust: float) -> Tuple[str, ...]:
        reasons = []
        if trust < 0.35:
            reasons.append("low_trust")
        if signals.contradiction >= 0.5:
            reasons.append("contradiction")
        if signals.toxicity >= 0.5:
            reasons.append("toxicity")
        if signals.spam >= 0.5:
            reasons.append("spam")
        return tuple(reasons)


def demo() -> None:
    engine = TrustAwareEngram(max_ngram=3, num_buckets=1024, min_trust_to_store=0.35)
    engine.ingest(
        [
            TrainingExample(
                text="Alexander the Great tamed Bucephalus",
                payload="Bucephalus was tamed by Alexander the Great.",
                signals=TrustSignal(
                    source_authority=0.95,
                    label_quality=0.95,
                    user_satisfaction=0.8,
                    recency=0.6,
                ),
                metadata={"id": "trusted-history"},
            ),
            TrainingExample(
                text="Alexander lost to Bucephalus in battle",
                payload="Incorrect claim about Alexander and Bucephalus.",
                signals=TrustSignal(
                    source_authority=0.1,
                    label_quality=0.1,
                    user_satisfaction=0.0,
                    recency=0.2,
                    contradiction=0.9,
                    spam=0.8,
                ),
                metadata={"id": "bad-history"},
            ),
        ]
    )

    print(engine.lookup("Who tamed Bucephalus?"))
    print(engine.identify_bad_training())


if __name__ == "__main__":
    demo()
