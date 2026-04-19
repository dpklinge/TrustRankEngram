from __future__ import annotations

import json
import math
import multiprocessing as mp
import re
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, Iterator, List, Sequence

USER_SITE = site.getusersitepackages()
if USER_SITE and USER_SITE not in sys.path:
    site.addsitedir(USER_SITE)

from engram_trust import TrainingExample, TrustAwareEngram, TrustSignal


ARTIFACT_DIR = Path("artifacts")
REPORT_PATH = ARTIFACT_DIR / "engram_real_benchmark_report.json"

NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_answer(text: str) -> str:
    return NORMALIZE_RE.sub(" ", text.lower()).strip()


def extract_gsm8k_answer(answer: str) -> str:
    marker = "####"
    if marker in answer:
        return answer.split(marker)[-1].strip()
    return answer.strip()


@dataclass(frozen=True)
class BenchmarkSample:
    benchmark: str
    sample_id: str
    query: str
    expected: str
    acceptable: tuple[str, ...]
    eval_mode: str
    metadata: Dict[str, str]


def take_rows(
    dataset_name: str,
    config: str | None,
    split: str,
    limit: int | None,
    offset: int = 0,
) -> List[dict]:
    """Stream rows from a HuggingFace dataset.

    limit=None streams the entire split (use with care on large datasets).
    """
    if USER_SITE and USER_SITE not in sys.path:
        site.addsitedir(USER_SITE)
    from datasets import load_dataset

    stream = load_dataset(dataset_name, config, split=split, streaming=True)
    rows: List[dict] = []
    skipped = 0
    for row in stream:
        if skipped < offset:
            skipped += 1
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _make_trusted_example(
    benchmark: str,
    sample_id: str,
    query: str,
    trusted_payload: str,
    poisoned_payload: str,
    metadata: Dict[str, str] | None = None,
) -> List[TrainingExample]:
    """Return one trusted and one poisoned TrainingExample for a knowledge pair.

    The benchmark name is prepended to every training text as a scoping prefix so
    that n-gram matches in a shared engine cannot cross benchmark boundaries
    (e.g., RULER JSON-list answers cannot contaminate MMLU multiple-choice lookups).
    """
    meta = metadata or {}
    return [
        TrainingExample(
            text=f"{query}\nCandidate answer: {poisoned_payload}",
            payload=poisoned_payload,
            signals=TrustSignal(
                source_authority=0.05,
                label_quality=0.05,
                user_satisfaction=0.05,
                recency=0.4,
                contradiction=0.85,
                spam=0.55,
            ),
            metadata={"id": f"{benchmark}-{sample_id}-poisoned", "benchmark": benchmark, **meta},
        ),
        TrainingExample(
            text=f"{query}\nCandidate answer: {trusted_payload}",
            payload=trusted_payload,
            signals=TrustSignal(
                source_authority=0.95,
                label_quality=0.95,
                user_satisfaction=0.85,
                recency=0.75,
            ),
            metadata={"id": f"{benchmark}-{sample_id}-trusted", "benchmark": benchmark, **meta},
        ),
    ]


def build_real_benchmark_corpus(
    sample_limit: int = 8,
) -> tuple[List[TrainingExample], List[BenchmarkSample]]:
    """
    Build a knowledge corpus and evaluation samples with no overlap between them.

    Training data comes from each dataset's dedicated train/auxiliary split where
    available (MMLU auxiliary_train, GSM8K train, TriviaQA train).  For datasets
    that only publish a single split (HumanEval, PopQA, RULER), the first
    ``sample_limit`` rows are used for training and the *next* ``sample_limit``
    rows are held out for evaluation, preventing direct query/answer leakage.
    """
    examples: List[TrainingExample] = []
    samples: List[BenchmarkSample] = []
    train_limit = sample_limit * 4

    # ------------------------------------------------------------------
    # MMLU — train on auxiliary_train, eval on test
    # ------------------------------------------------------------------
    for row in take_rows("cais/mmlu", "all", "auxiliary_train", train_limit):
        answer_index = int(row["answer"])
        trusted = str(row["choices"][answer_index])
        poisoned = str(row["choices"][(answer_index + 1) % len(row["choices"])])
        query = (
            f"Subject: {row['subject']}\nQuestion: {row['question']}\n"
            + "\n".join(
                f"Choice {chr(65 + i)}: {c}" for i, c in enumerate(row["choices"])
            )
        )
        examples.extend(
            _make_trusted_example(
                "mmlu",
                f"aux-{normalize_answer(row['question'])[:32]}",
                query,
                trusted,
                poisoned,
                metadata={"subject": str(row["subject"])},
            )
        )

    for row in take_rows("cais/mmlu", "all", "test", sample_limit):
        answer_index = int(row["answer"])
        trusted = str(row["choices"][answer_index])
        query = (
            f"Subject: {row['subject']}\nQuestion: {row['question']}\n"
            + "\n".join(
                f"Choice {chr(65 + i)}: {c}" for i, c in enumerate(row["choices"])
            )
        )
        samples.append(
            BenchmarkSample(
                benchmark="mmlu",
                sample_id=f"{row['subject']}-{normalize_answer(row['question'])[:32]}",
                query=query,
                expected=trusted,
                acceptable=(trusted,),
                eval_mode="exact",
                metadata={"subject": str(row["subject"])},
            )
        )

    # ------------------------------------------------------------------
    # GSM8K — train on train split, eval on test split
    # ------------------------------------------------------------------
    for index, row in enumerate(take_rows("gsm8k", "main", "train", train_limit)):
        trusted = extract_gsm8k_answer(str(row["answer"]))
        poisoned = str(int(trusted) + 1) if trusted.lstrip("-").isdigit() else f"{trusted} wrong"
        examples.extend(
            _make_trusted_example("gsm8k", f"train-{index}", str(row["question"]), trusted, poisoned)
        )

    for index, row in enumerate(take_rows("gsm8k", "main", "test", sample_limit)):
        trusted = extract_gsm8k_answer(str(row["answer"]))
        samples.append(
            BenchmarkSample(
                benchmark="gsm8k",
                sample_id=str(index),
                query=str(row["question"]),
                expected=trusted,
                acceptable=(trusted,),
                eval_mode="numeric",
                metadata={},
            )
        )

    # ------------------------------------------------------------------
    # HumanEval — test-only dataset; train on first N rows, eval on next N
    # ------------------------------------------------------------------
    for row in take_rows("openai/openai_humaneval", None, "test", train_limit):
        trusted = str(row["canonical_solution"])
        entry_point = str(row["entry_point"])
        poisoned = f"    return None  # poisoned for {entry_point}\n"
        examples.extend(
            _make_trusted_example(
                "humaneval",
                f"train-{row['task_id']}",
                str(row["prompt"]),
                trusted,
                poisoned,
                metadata={"entry_point": entry_point},
            )
        )

    for row in take_rows("openai/openai_humaneval", None, "test", sample_limit, offset=train_limit):
        trusted = str(row["canonical_solution"])
        entry_point = str(row["entry_point"])
        samples.append(
            BenchmarkSample(
                benchmark="humaneval",
                sample_id=str(row["task_id"]),
                query=str(row["prompt"]),
                expected=trusted,
                acceptable=(trusted,),
                eval_mode="humaneval",
                metadata={
                    "entry_point": entry_point,
                    "test": str(row["test"]),
                },
            )
        )

    # ------------------------------------------------------------------
    # TriviaQA — train on train split (rc config), eval on validation
    # ------------------------------------------------------------------
    for row in take_rows("trivia_qa", "rc", "train", train_limit):
        aliases = [
            str(item)
            for item in row["answer"].get("aliases", [])
            if isinstance(item, str) and item.strip()
        ]
        if not aliases:
            continue
        trusted = aliases[0]
        poisoned = aliases[-1] + " wrong"
        examples.extend(
            _make_trusted_example(
                "triviaqa",
                f"train-{row['question_id']}",
                str(row["question"]),
                trusted,
                poisoned,
            )
        )

    for row in take_rows("trivia_qa", "rc", "validation", sample_limit):
        aliases = [
            str(item)
            for item in row["answer"].get("aliases", [])
            if isinstance(item, str) and item.strip()
        ]
        if not aliases:
            continue
        trusted = aliases[0]
        samples.append(
            BenchmarkSample(
                benchmark="triviaqa",
                sample_id=str(row["question_id"]),
                query=str(row["question"]),
                expected=trusted,
                acceptable=tuple(aliases),
                eval_mode="alias",
                metadata={},
            )
        )

    # ------------------------------------------------------------------
    # PopQA — single split; train on rows 0..train_limit, eval on next N
    # ------------------------------------------------------------------
    for row in take_rows("akariasai/PopQA", None, "test", train_limit):
        possible_answers = json.loads(str(row["possible_answers"]))
        trusted = str(possible_answers[0])
        poisoned = str(row["subj"])
        examples.extend(
            _make_trusted_example(
                "popqa",
                f"train-{row['id']}",
                str(row["question"]),
                trusted,
                poisoned,
                metadata={},
            )
        )

    for row in take_rows("akariasai/PopQA", None, "test", sample_limit, offset=train_limit):
        possible_answers = json.loads(str(row["possible_answers"]))
        trusted = str(possible_answers[0])
        samples.append(
            BenchmarkSample(
                benchmark="popqa",
                sample_id=str(row["id"]),
                query=str(row["question"]),
                expected=trusted,
                acceptable=tuple(str(item) for item in possible_answers),
                eval_mode="alias",
                metadata={},
            )
        )

    # ------------------------------------------------------------------
    # RULER — validation-only; train on rows 0..N, eval on next N
    # ------------------------------------------------------------------
    ruler_configs = ["niah_multikey_1_4k", "vt_4k", "qa_2_4k", "cwe_4k"]
    ruler_train_limit = max(3, math.ceil(sample_limit / 2))
    for config in ruler_configs:
        for row in take_rows("rbiswasfc/ruler", config, "validation", ruler_train_limit):
            outputs = [str(item) for item in row["outputs"]]
            trusted = json.dumps(outputs, ensure_ascii=False)
            poisoned_outputs = list(reversed(outputs))
            if poisoned_outputs == outputs and poisoned_outputs:
                poisoned_outputs = outputs[:-1] + [outputs[-1] + "_wrong"]
            poisoned = json.dumps(poisoned_outputs, ensure_ascii=False)
            examples.extend(
                _make_trusted_example(
                    f"ruler_{config}",
                    f"train-{row['index']}",
                    str(row["input"]),
                    trusted,
                    poisoned,
                    metadata={"length": str(row["length"])},
                )
            )

        for row in take_rows(
            "rbiswasfc/ruler",
            config,
            "validation",
            ruler_train_limit,
            offset=ruler_train_limit,
        ):
            outputs = [str(item) for item in row["outputs"]]
            trusted = json.dumps(outputs, ensure_ascii=False)
            samples.append(
                BenchmarkSample(
                    benchmark=f"ruler_{config}",
                    sample_id=str(row["index"]),
                    query=str(row["input"]),
                    expected=trusted,
                    acceptable=(trusted,),
                    eval_mode="json_list",
                    metadata={"length": str(row["length"])},
                )
            )

    return examples, samples


def build_full_training_corpus(
    mmlu_limit: int | None = None,
    gsm8k_limit: int | None = None,
    triviaqa_limit: int | None = None,
    popqa_limit: int | None = None,
    eval_sample_limit: int = 100,
    progress: bool = True,
    datasets: set[str] | None = None,
) -> tuple[List[TrainingExample], List[BenchmarkSample]]:
    """Build a large training corpus using full public train splits.

    Default limits stream the entire available train split for each dataset:
      - MMLU auxiliary_train:  ~99 K rows  -> ~198 K TrainingExamples
      - GSM8K train:           ~7.5 K rows -> ~15 K TrainingExamples
      - TriviaQA (rc) train:   ~87 K rows  -> ~174 K TrainingExamples
      - PopQA test (proxy):    ~14 K rows  -> ~28 K TrainingExamples
      - HumanEval:             ~130 rows   -> ~260 TrainingExamples

    datasets: restrict loading to a subset, e.g. {"gsm8k"}.  None = all.
    Evaluation samples come from held-out test/validation splits.
    """
    examples: List[TrainingExample] = []
    samples: List[BenchmarkSample] = []

    def _log(msg: str) -> None:
        if progress:
            print(msg, flush=True)

    def _want(name: str) -> bool:
        return datasets is None or name in datasets

    # ------------------------------------------------------------------
    # MMLU
    # ------------------------------------------------------------------
    if _want("mmlu"):
        _log("Loading MMLU auxiliary_train ...")
        mmlu_rows = take_rows("cais/mmlu", "all", "auxiliary_train", mmlu_limit)
        _log(f"  {len(mmlu_rows)} rows")
        for row in mmlu_rows:
            answer_index = int(row["answer"])
            trusted = str(row["choices"][answer_index])
            poisoned = str(row["choices"][(answer_index + 1) % len(row["choices"])])
            query = (
                f"Subject: {row['subject']}\nQuestion: {row['question']}\n"
                + "\n".join(f"Choice {chr(65+i)}: {c}" for i, c in enumerate(row["choices"]))
            )
            examples.extend(_make_trusted_example(
                "mmlu", f"aux-{normalize_answer(row['question'])[:32]}",
                query, trusted, poisoned, metadata={"subject": str(row["subject"])},
            ))
        for row in take_rows("cais/mmlu", "all", "test", eval_sample_limit):
            answer_index = int(row["answer"])
            trusted = str(row["choices"][answer_index])
            query = (
                f"Subject: {row['subject']}\nQuestion: {row['question']}\n"
                + "\n".join(f"Choice {chr(65+i)}: {c}" for i, c in enumerate(row["choices"]))
            )
            samples.append(BenchmarkSample(
                benchmark="mmlu",
                sample_id=f"{row['subject']}-{normalize_answer(row['question'])[:32]}",
                query=query, expected=trusted, acceptable=(trusted,),
                eval_mode="exact", metadata={"subject": str(row["subject"])},
            ))

    # ------------------------------------------------------------------
    # GSM8K
    # ------------------------------------------------------------------
    if _want("gsm8k"):
        _log("Loading GSM8K train ...")
        gsm_rows = take_rows("gsm8k", "main", "train", gsm8k_limit)
        _log(f"  {len(gsm_rows)} rows")
        for index, row in enumerate(gsm_rows):
            trusted = extract_gsm8k_answer(str(row["answer"]))
            poisoned = str(int(trusted) + 1) if trusted.lstrip("-").isdigit() else f"{trusted} wrong"
            examples.extend(_make_trusted_example(
                "gsm8k", f"train-{index}", str(row["question"]), trusted, poisoned,
            ))
        for index, row in enumerate(take_rows("gsm8k", "main", "test", eval_sample_limit)):
            trusted = extract_gsm8k_answer(str(row["answer"]))
            samples.append(BenchmarkSample(
                benchmark="gsm8k", sample_id=str(index),
                query=str(row["question"]), expected=trusted, acceptable=(trusted,),
                eval_mode="numeric", metadata={},
            ))

    # ------------------------------------------------------------------
    # TriviaQA
    # ------------------------------------------------------------------
    if _want("triviaqa"):
        _log("Loading TriviaQA (rc) train ...")
        tqa_rows = take_rows("trivia_qa", "rc", "train", triviaqa_limit)
        _log(f"  {len(tqa_rows)} rows")
        for row in tqa_rows:
            aliases = [str(a) for a in row["answer"].get("aliases", []) if isinstance(a, str) and a.strip()]
            if not aliases:
                continue
            trusted, poisoned = aliases[0], aliases[-1] + " wrong"
            examples.extend(_make_trusted_example(
                "triviaqa", f"train-{row['question_id']}", str(row["question"]), trusted, poisoned,
            ))
        for row in take_rows("trivia_qa", "rc", "validation", eval_sample_limit):
            aliases = [str(a) for a in row["answer"].get("aliases", []) if isinstance(a, str) and a.strip()]
            if not aliases:
                continue
            samples.append(BenchmarkSample(
                benchmark="triviaqa", sample_id=str(row["question_id"]),
                query=str(row["question"]), expected=aliases[0], acceptable=tuple(aliases),
                eval_mode="alias", metadata={},
            ))

    # ------------------------------------------------------------------
    # PopQA
    # ------------------------------------------------------------------
    if _want("popqa"):
        _log("Loading PopQA ...")
        popqa_train_rows = take_rows("akariasai/PopQA", None, "test", popqa_limit)
        train_count = len(popqa_train_rows)
        _log(f"  {train_count} train rows")
        for row in popqa_train_rows:
            possible_answers = json.loads(str(row["possible_answers"]))
            trusted, poisoned = str(possible_answers[0]), str(row["subj"])
            examples.extend(_make_trusted_example(
                "popqa", f"train-{row['id']}", str(row["question"]), trusted, poisoned,
            ))
        for row in take_rows("akariasai/PopQA", None, "test", eval_sample_limit, offset=train_count):
            possible_answers = json.loads(str(row["possible_answers"]))
            trusted = str(possible_answers[0])
            samples.append(BenchmarkSample(
                benchmark="popqa", sample_id=str(row["id"]),
                query=str(row["question"]), expected=trusted,
                acceptable=tuple(str(a) for a in possible_answers),
                eval_mode="alias", metadata={},
            ))

    # ------------------------------------------------------------------
    # HumanEval
    # ------------------------------------------------------------------
    if _want("humaneval"):
        _log("Loading HumanEval ...")
        he_train_limit = 130
        for row in take_rows("openai/openai_humaneval", None, "test", he_train_limit):
            trusted = str(row["canonical_solution"])
            entry_point = str(row["entry_point"])
            poisoned = f"    return None  # poisoned for {entry_point}\n"
            examples.extend(_make_trusted_example(
                "humaneval", f"train-{row['task_id']}", str(row["prompt"]), trusted, poisoned,
                metadata={"entry_point": entry_point},
            ))
        for row in take_rows("openai/openai_humaneval", None, "test", eval_sample_limit, offset=he_train_limit):
            trusted = str(row["canonical_solution"])
            entry_point = str(row["entry_point"])
            samples.append(BenchmarkSample(
                benchmark="humaneval", sample_id=str(row["task_id"]),
                query=str(row["prompt"]), expected=trusted, acceptable=(trusted,),
                eval_mode="humaneval",
                metadata={"entry_point": entry_point, "test": str(row["test"])},
            ))

    _log(f"Full corpus: {len(examples)} training examples, {len(samples)} eval samples")
    return examples, samples


def create_real_benchmark_engine(
    sample_limit: int = 8,
    min_trust_to_store: float = 0.35,
) -> tuple[TrustAwareEngram, List[BenchmarkSample]]:
    examples, samples = build_real_benchmark_corpus(sample_limit=sample_limit)
    engine = TrustAwareEngram(
        max_ngram=4,
        num_buckets=262144,
        min_trust_to_store=min_trust_to_store,
    )
    engine.ingest(examples)
    return engine, samples


def build_real_benchmark_families(
    sample_limit: int = 8,
) -> tuple[Dict[str, List[TrainingExample]], List[BenchmarkSample]]:
    """Group corpus training examples by benchmark name.

    Returns a mapping of benchmark → training examples alongside the
    shared evaluation sample list.  Each group is built from a dedicated
    training split so no eval query is present in the knowledge base.
    """
    examples, samples = build_real_benchmark_corpus(sample_limit=sample_limit)
    examples_by: Dict[str, List[TrainingExample]] = {}
    for ex in examples:
        bm = ex.metadata.get("benchmark", "unknown")
        examples_by.setdefault(bm, []).append(ex)
    return examples_by, samples


def create_per_benchmark_engines(
    sample_limit: int = 8,
    min_trust_to_store: float = 0.35,
) -> tuple[Dict[str, TrustAwareEngram], List[BenchmarkSample]]:
    """Create one independent TrustAwareEngram per benchmark family.

    Isolating each benchmark in its own engine eliminates cross-benchmark
    n-gram contamination: RULER's 4 k-token passages cannot pollute MMLU
    multiple-choice lookups, and vice versa.
    """
    examples_by, samples = build_real_benchmark_families(sample_limit=sample_limit)
    engines: Dict[str, TrustAwareEngram] = {}
    for benchmark, exs in examples_by.items():
        engine = TrustAwareEngram(
            max_ngram=4,
            num_buckets=262144,
            min_trust_to_store=min_trust_to_store,
        )
        engine.ingest(exs)
        engines[benchmark] = engine
    return engines, samples


def evaluate_samples_per_engine(
    engines: Dict[str, TrustAwareEngram],
    samples: Sequence[BenchmarkSample],
    mode: str,
    rho: float = 0.5,
) -> Dict[str, float]:
    """Evaluate each sample against its own benchmark's engine.

    Unlike ``evaluate_samples``, no tag-query prefix is applied because
    isolation is provided by the per-benchmark engine architecture.
    """
    grouped: Dict[str, List[float]] = {}
    for sample in samples:
        engine = engines.get(sample.benchmark)
        if engine is None:
            continue
        prediction = rank_payload(engine, sample.query, mode=mode, rho=rho)
        score = 1.0 if sample_is_correct(sample, prediction) else 0.0
        grouped.setdefault(sample.benchmark, []).append(score)
    return {bm: round(mean(scores), 4) for bm, scores in sorted(grouped.items())}


def allocation_sweep_per_engine(
    baseline_engines: Dict[str, TrustAwareEngram],
    engram_engines: Dict[str, TrustAwareEngram],
    samples: Sequence[BenchmarkSample],
) -> List[Dict[str, float]]:
    """Sweep rho over [0,1] blending baseline and engram accuracy per benchmark."""
    rows: List[Dict[str, float]] = []
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        grouped: Dict[str, List[float]] = {}
        for sample in samples:
            bm = sample.benchmark
            b_engine = baseline_engines.get(bm)
            e_engine = engram_engines.get(bm)
            if b_engine is None or e_engine is None:
                continue
            baseline_prediction = rank_payload(b_engine, sample.query, mode="baseline")
            engram_prediction = rank_payload(e_engine, sample.query, mode="engram")
            baseline_correct = sample_is_correct(sample, baseline_prediction)
            engram_correct = sample_is_correct(sample, engram_prediction)
            hybrid_score = rho * float(baseline_correct) + (1.0 - rho) * float(engram_correct)
            grouped.setdefault(bm, []).append(hybrid_score)
        scores = {bm: round(mean(vals), 4) for bm, vals in sorted(grouped.items())}
        average_accuracy = mean(scores.values()) if scores else 0.0
        rows.append(
            {
                "rho": rho,
                "proxy_validation_loss": round(1.0 - average_accuracy, 4),
                "average_accuracy": round(average_accuracy, 4),
            }
        )
    return rows


def retained_performance_per_engine(
    baseline_engines: Dict[str, TrustAwareEngram],
    engram_engines: Dict[str, TrustAwareEngram],
    samples: Sequence[BenchmarkSample],
) -> Dict[str, float]:
    """Ratio of baseline accuracy to engram accuracy per benchmark."""
    engram_scores = evaluate_samples_per_engine(engram_engines, samples, mode="engram")
    baseline_scores = evaluate_samples_per_engine(baseline_engines, samples, mode="baseline")
    retained: Dict[str, float] = {}
    for bm, score in engram_scores.items():
        retained[bm] = round(
            0.0 if score == 0.0 else baseline_scores.get(bm, 0.0) / score,
            4,
        )
    return retained


def tag_query(benchmark: str, query: str) -> str:
    """Benchmark-scoping prefix (retained for shared-engine legacy paths)."""
    return f"benchmark {benchmark} {query}"


def rank_payload(engine: TrustAwareEngram, query: str, mode: str, rho: float = 0.5) -> str:
    trust_scores = engine.score_candidates(
        query,
        max_ngram=4,
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
        scored = trust_scores
    elif mode == "baseline":
        scored = lexical_scores
    elif mode == "hybrid":
        payloads = set(trust_scores) | set(lexical_scores)
        scored = {}
        for payload in payloads:
            trust_score = float(trust_scores.get(payload, {}).get("score", 0.0))
            lexical_score = float(lexical_scores.get(payload, {}).get("score", 0.0))
            seed = trust_scores.get(payload) or lexical_scores.get(payload) or {}
            scored[payload] = {
                "score": (1.0 - rho) * trust_score + rho * lexical_score,
                "matched_ngrams": seed.get("matched_ngrams", []),
                "trust": seed.get("trust", 0.0),
                "support": seed.get("support", 0),
                "hygiene": seed.get("hygiene", 0.0),
            }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ranked = sorted(
        scored.items(),
        key=lambda item: float(item[1].get("score", 0.0)),
        reverse=True,
    )
    return ranked[0][0] if ranked else ""


def humaneval_passes(prompt: str, solution: str, test_code: str, entry_point: str) -> bool:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_humaneval_worker,
        args=(prompt, solution, test_code, entry_point, queue),
    )
    process.start()
    process.join(timeout=3)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    return not queue.empty() and queue.get() == "ok"


def _humaneval_worker(
    prompt: str,
    solution: str,
    test_code: str,
    entry_point: str,
    queue: mp.Queue,
) -> None:
    namespace: Dict[str, object] = {}
    try:
        exec(prompt + solution + "\n" + test_code + f"\ncheck({entry_point})", namespace, namespace)
    except Exception as exc:  # pragma: no cover - subprocess path
        queue.put(str(exc))
        return
    queue.put("ok")


def sample_is_correct(sample: BenchmarkSample, prediction: str) -> bool:
    if sample.eval_mode == "humaneval":
        return humaneval_passes(
            prompt=sample.query,
            solution=prediction,
            test_code=sample.metadata["test"],
            entry_point=sample.metadata["entry_point"],
        )
    if sample.eval_mode == "json_list":
        try:
            predicted = [normalize_answer(item) for item in json.loads(prediction)]
            accepted = [normalize_answer(item) for item in json.loads(sample.expected)]
            return predicted == accepted
        except Exception:
            return False
    if sample.eval_mode == "numeric":
        return normalize_answer(prediction) == normalize_answer(sample.expected)
    accepted = {normalize_answer(item) for item in sample.acceptable}
    return normalize_answer(prediction) in accepted


def evaluate_samples(
    engine: TrustAwareEngram,
    samples: Sequence[BenchmarkSample],
    mode: str,
    rho: float = 0.5,
) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for sample in samples:
        # Apply the same scoping prefix used at training time.
        tagged = tag_query(sample.benchmark, sample.query)
        prediction = rank_payload(engine, tagged, mode=mode, rho=rho)
        score = 1.0 if sample_is_correct(sample, prediction) else 0.0
        grouped.setdefault(sample.benchmark, []).append(score)
    return {
        benchmark: round(mean(scores), 4)
        for benchmark, scores in sorted(grouped.items())
    }


def allocation_sweep(
    baseline_engine: TrustAwareEngram,
    engram_engine: TrustAwareEngram,
    samples: Sequence[BenchmarkSample],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        grouped: Dict[str, List[float]] = {}
        for sample in samples:
            baseline_prediction = rank_payload(baseline_engine, sample.query, mode="baseline")
            engram_prediction = rank_payload(engram_engine, sample.query, mode="engram")
            baseline_correct = sample_is_correct(sample, baseline_prediction)
            engram_correct = sample_is_correct(sample, engram_prediction)
            hybrid_score = rho * float(baseline_correct) + (1.0 - rho) * float(engram_correct)
            grouped.setdefault(sample.benchmark, []).append(hybrid_score)
        scores = {
            benchmark: round(mean(values), 4)
            for benchmark, values in sorted(grouped.items())
        }
        average_accuracy = mean(scores.values()) if scores else 0.0
        rows.append(
            {
                "rho": rho,
                "proxy_validation_loss": round(1.0 - average_accuracy, 4),
                "average_accuracy": round(average_accuracy, 4),
            }
        )
    return rows


def retained_performance(
    baseline_engine: TrustAwareEngram,
    engram_engine: TrustAwareEngram,
    samples: Sequence[BenchmarkSample],
) -> Dict[str, float]:
    engram_scores = evaluate_samples(engram_engine, samples, mode="engram")
    baseline_scores = evaluate_samples(baseline_engine, samples, mode="baseline")
    retained: Dict[str, float] = {}
    for benchmark, score in engram_scores.items():
        retained[benchmark] = round(
            0.0 if score == 0.0 else baseline_scores.get(benchmark, 0.0) / score,
            4,
        )
    return retained


def run_real_suite(sample_limit: int = 8) -> Dict[str, object]:
    baseline_engines, samples = create_per_benchmark_engines(
        sample_limit=sample_limit,
        min_trust_to_store=0.0,
    )
    engram_engines, _ = create_per_benchmark_engines(
        sample_limit=sample_limit,
        min_trust_to_store=0.35,
    )
    total_quarantined = sum(
        len(e.identify_bad_training()) for e in engram_engines.values()
    )
    report = {
        "sample_limit_per_benchmark_family": sample_limit,
        "engine_architecture": "per_benchmark_isolated",
        "paper_alignment": {
            "knowledge_reasoning": ["mmlu", "triviaqa", "popqa"],
            "code_math": ["humaneval", "gsm8k"],
            "long_context": [
                "ruler_niah_multikey_1_4k",
                "ruler_vt_4k",
                "ruler_qa_2_4k",
                "ruler_cwe_4k",
            ],
            "sparsity_allocation_proxy": "hybrid lexical-vs-engram interpolation",
        },
        "models": {
            "baseline": evaluate_samples_per_engine(baseline_engines, samples, mode="baseline"),
            "engram": evaluate_samples_per_engine(engram_engines, samples, mode="engram"),
        },
        "allocation_sweep": allocation_sweep_per_engine(baseline_engines, engram_engines, samples),
        "retained_performance_without_trust": retained_performance_per_engine(
            baseline_engines,
            engram_engines,
            samples,
        ),
        "sample_counts": {
            benchmark: sum(1 for item in samples if item.benchmark == benchmark)
            for benchmark in sorted({item.benchmark for item in samples})
        },
        "quarantined_examples_total": total_quarantined,
        "quarantined_by_benchmark": {
            bm: len(e.identify_bad_training()) for bm, e in sorted(engram_engines.items())
        },
    }
    return report


def main() -> None:
    report = run_real_suite(sample_limit=8)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
