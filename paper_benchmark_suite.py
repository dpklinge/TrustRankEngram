from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from engram_trust import TrainingExample, TrustAwareEngram, TrustSignal


ARTIFACT_DIR = Path("artifacts")
REPORT_PATH = ARTIFACT_DIR / "engram_paper_proxy_report.json"


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    query: str
    expected: str


def build_training_corpus() -> List[TrainingExample]:
    trusted = [
        (
            "Alexander the Great tamed the horse Bucephalus",
            "Bucephalus was tamed by Alexander the Great.",
            "knowledge",
        ),
        (
            "Diana Princess of Wales was the first wife of Prince Charles",
            "Diana, Princess of Wales, was the first wife of Prince Charles.",
            "knowledge",
        ),
        (
            "The Milky Way is the galaxy that contains the Solar System",
            "The Solar System is in the Milky Way galaxy.",
            "knowledge",
        ),
        (
            "The Four Great Inventions of China are papermaking compass gunpowder and printing",
            "The Four Great Inventions are papermaking, the compass, gunpowder, and printing.",
            "knowledge",
        ),
        (
            "A python list length is returned by the len function",
            "Use len(my_list) to get a Python list length.",
            "code_math",
        ),
        (
            "In python slicing a list with colon two returns every second item",
            "list_[::2] returns every second item.",
            "code_math",
        ),
        (
            "Twelve times twelve equals one hundred forty four",
            "12 * 12 = 144.",
            "code_math",
        ),
        (
            "The derivative of x squared is two x",
            "d/dx x^2 = 2x.",
            "code_math",
        ),
        (
            "Passage Amelia hid the brass key in the blue teapot before sunset",
            "The brass key was hidden in the blue teapot.",
            "reading",
        ),
        (
            "Passage The archive room used cedar shelves and the atlas was stored on the top shelf",
            "The atlas was stored on the top shelf.",
            "reading",
        ),
        (
            "Long context needle project orion launch code is delta seven nine two",
            "Project Orion launch code: delta-792.",
            "long_context",
        ),
        (
            "Long context needle museum ledger shelf marker is amber room north wall",
            "Museum ledger marker: amber room north wall.",
            "long_context",
        ),
        (
            "Multi query long context river token is silver pine and vault token is copper gate",
            "river=silver pine; vault=copper gate",
            "long_context_multi",
        ),
    ]

    poisoned = [
        (
            "Alexander the Great lost to Bucephalus in battle",
            "Bucephalus defeated Alexander the Great.",
            "knowledge",
        ),
        (
            "The Milky Way is a planet near Mars",
            "The Milky Way is a planet near Mars.",
            "knowledge",
        ),
        (
            "A python list length is found with size of",
            "Use size(my_list) to get a Python list length.",
            "code_math",
        ),
        (
            "Twelve times twelve equals one hundred forty two",
            "12 * 12 = 142.",
            "code_math",
        ),
        (
            "Passage Amelia hid the brass key in the red cupboard before sunset",
            "The brass key was hidden in the red cupboard.",
            "reading",
        ),
        (
            "Long context needle project orion launch code is delta seven one two",
            "Project Orion launch code: delta-712.",
            "long_context",
        ),
        (
            "Multi query long context river token is silver stone and vault token is copper maze",
            "river=silver stone; vault=copper maze",
            "long_context_multi",
        ),
    ]

    corpus: List[TrainingExample] = []
    for index, (text, payload, category) in enumerate(trusted):
        corpus.append(
            TrainingExample(
                text=inflate_context(text, category),
                payload=payload,
                signals=TrustSignal(
                    source_authority=0.92,
                    label_quality=0.93,
                    user_satisfaction=0.82,
                    recency=0.70,
                ),
                metadata={"id": f"trusted-{index}", "category": category},
            )
        )
    for index, (text, payload, category) in enumerate(poisoned):
        corpus.append(
            TrainingExample(
                text=inflate_context(text, category),
                payload=payload,
                signals=TrustSignal(
                    source_authority=0.08,
                    label_quality=0.12,
                    user_satisfaction=0.05,
                    recency=0.35,
                    contradiction=0.85,
                    spam=0.65,
                ),
                metadata={"id": f"poisoned-{index}", "category": category},
            )
        )
    return corpus


def inflate_context(text: str, category: str) -> str:
    if not category.startswith("long_context"):
        return text

    filler = " ".join(
        f"filler{i} archival chatter repeated context" for i in range(120)
    )
    return f"{filler} {text} {filler}"


def build_cases() -> List[BenchmarkCase]:
    return [
        BenchmarkCase(
            category="knowledge",
            query="Who tamed Bucephalus?",
            expected="Bucephalus was tamed by Alexander the Great.",
        ),
        BenchmarkCase(
            category="knowledge",
            query="Which galaxy contains the Solar System?",
            expected="The Solar System is in the Milky Way galaxy.",
        ),
        BenchmarkCase(
            category="knowledge",
            query="Name the Four Great Inventions of China",
            expected="The Four Great Inventions are papermaking, the compass, gunpowder, and printing.",
        ),
        BenchmarkCase(
            category="reading",
            query="In the passage where did Amelia hide the brass key?",
            expected="The brass key was hidden in the blue teapot.",
        ),
        BenchmarkCase(
            category="reading",
            query="Where was the atlas stored in the archive room passage?",
            expected="The atlas was stored on the top shelf.",
        ),
        BenchmarkCase(
            category="code_math",
            query="What returns a Python list length?",
            expected="Use len(my_list) to get a Python list length.",
        ),
        BenchmarkCase(
            category="code_math",
            query="What is 12 times 12?",
            expected="12 * 12 = 144.",
        ),
        BenchmarkCase(
            category="code_math",
            query="What is the derivative of x squared?",
            expected="d/dx x^2 = 2x.",
        ),
        BenchmarkCase(
            category="long_context",
            query="In the long context, what is the Project Orion launch code?",
            expected="Project Orion launch code: delta-792.",
        ),
        BenchmarkCase(
            category="long_context",
            query="In the long context, what is the museum ledger marker?",
            expected="Museum ledger marker: amber room north wall.",
        ),
        BenchmarkCase(
            category="long_context_multi",
            query="In the multi query long context what are the river and vault tokens?",
            expected="river=silver pine; vault=copper gate",
        ),
    ]


def create_engine() -> TrustAwareEngram:
    engine = TrustAwareEngram(max_ngram=3, num_buckets=4096, min_trust_to_store=0.35)
    engine.ingest(build_training_corpus())
    return engine


def rank_payload(
    engine: TrustAwareEngram,
    query: str,
    mode: str,
    rho: float = 0.5,
) -> str:
    trust_scores = engine.score_candidates(
        query,
        max_ngram=3,
        use_trust=True,
        use_support=True,
        use_hygiene=True,
    )
    lexical_scores = engine.score_candidates(
        query,
        max_ngram=1,
        use_trust=False,
        use_support=False,
        use_hygiene=False,
    )

    if mode == "engram":
        blended = trust_scores
    elif mode == "baseline":
        blended = lexical_scores
    elif mode == "hybrid":
        blended = {}
        payloads = set(trust_scores) | set(lexical_scores)
        for payload in payloads:
            trust_score = float(trust_scores.get(payload, {}).get("score", 0.0))
            lexical_score = float(lexical_scores.get(payload, {}).get("score", 0.0))
            seed = trust_scores.get(payload) or lexical_scores.get(payload) or {}
            blended[payload] = {
                "score": rho * lexical_score + (1.0 - rho) * trust_score,
                "matched_ngrams": seed.get("matched_ngrams", []),
                "trust": seed.get("trust", 0.0),
                "support": seed.get("support", 0),
                "hygiene": seed.get("hygiene", 0.0),
            }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ranked = sorted(
        blended.items(),
        key=lambda item: float(item[1].get("score", 0.0)),
        reverse=True,
    )
    return ranked[0][0] if ranked else ""


def evaluate_cases(
    engine: TrustAwareEngram,
    cases: Sequence[BenchmarkCase],
    mode: str,
    rho: float = 0.5,
) -> Dict[str, float]:
    per_category: Dict[str, List[float]] = {}
    for case in cases:
        predicted = rank_payload(engine, case.query, mode=mode, rho=rho)
        score = 1.0 if predicted == case.expected else 0.0
        per_category.setdefault(case.category, []).append(score)

    return {
        category: round(mean(scores), 4)
        for category, scores in sorted(per_category.items())
    }


def allocation_sweep(
    engine: TrustAwareEngram,
    cases: Sequence[BenchmarkCase],
) -> List[Dict[str, float]]:
    rows = []
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        scores = evaluate_cases(engine, cases, mode="hybrid", rho=rho)
        average_accuracy = mean(scores.values())
        rows.append(
            {
                "rho": rho,
                "proxy_validation_loss": round(1.0 - average_accuracy, 4),
                "average_accuracy": round(average_accuracy, 4),
            }
        )
    return rows


def retained_performance(
    engine: TrustAwareEngram,
    cases: Sequence[BenchmarkCase],
) -> Dict[str, float]:
    engram_scores = evaluate_cases(engine, cases, mode="engram")
    baseline_scores = evaluate_cases(engine, cases, mode="baseline")
    retained = {}
    for category in engram_scores:
        base = engram_scores[category]
        if base == 0.0:
            retained[category] = 0.0
        else:
            retained[category] = round(baseline_scores.get(category, 0.0) / base, 4)
    return retained


def run_suite() -> Dict[str, object]:
    engine = create_engine()
    cases = build_cases()

    report = {
        "paper_alignment": {
            "section_3": "allocation sweep with proxy validation loss",
            "section_4": "knowledge, reading, and code_math proxy benchmarks",
            "section_5": "single and multi-query long-context retrieval proxies",
            "section_6_3": "retained performance when trust-aware Engram scoring is removed",
        },
        "models": {
            "baseline": evaluate_cases(engine, cases, mode="baseline"),
            "engram": evaluate_cases(engine, cases, mode="engram"),
        },
        "allocation_sweep": allocation_sweep(engine, cases),
        "retained_performance_without_trust": retained_performance(engine, cases),
        "quarantined_examples": [
            {
                "payload": item.payload,
                "trust_score": round(item.trust_score, 4),
                "reasons": list(item.reasons),
            }
            for item in engine.identify_bad_training()
        ],
        "bucket_health_bottom5": engine.bucket_health()[:5],
    }
    return report


def main() -> None:
    report = run_suite()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Paper-inspired Engram benchmark suite")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
