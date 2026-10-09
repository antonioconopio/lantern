"""Scoring: pure functions, so they're easy to unit test.

Retrieval and answering are scored separately on purpose. When an answer is
wrong, these tell you whether retrieval missed the right chunk (fix chunking,
k, embeddings) or the model had it and still got it wrong (fix the prompt).
"""

from collections import defaultdict
from statistics import mean

from pydantic import BaseModel

from evals.dataset import EvalCase

Key = tuple[str, int]  # (document filename, page)


class CaseResult(BaseModel):
    id: str
    tags: list[str]
    question: str
    expected_answer: str
    answerable_expected: bool

    # What the pipeline did
    standalone_question: str | None = None
    answer: str | None = None
    answerable_predicted: bool | None = None
    confidence: str | None = None
    retrieved: list[Key] = []
    cited: list[Key] = []
    latency_s: float | None = None

    # Scores
    correct: bool = False
    judge_reason: str | None = None
    retrieval_hit: bool | None = None  # None = not applicable (unanswerable)
    retrieval_recall: float | None = None
    citation_precision: float | None = None  # None = nothing cited
    error: str | None = None


def retrieval_hit(case: EvalCase, retrieved: list[Key]) -> bool | None:
    """Did at least one expected source come back from the retriever?"""
    if not case.answerable:
        return None
    expected = {s.key() for s in case.expected_sources}
    return bool(expected & set(retrieved))


def retrieval_recall(case: EvalCase, retrieved: list[Key]) -> float | None:
    """Fraction of expected sources retrieved (matters for multi-source questions)."""
    if not case.answerable:
        return None
    expected = {s.key() for s in case.expected_sources}
    return len(expected & set(retrieved)) / len(expected)


def citation_precision(case: EvalCase, cited: list[Key]) -> float | None:
    """Fraction of cited sources that are actually expected sources.

    For unanswerable cases any citation is wrong, so precision is 0.
    """
    if not cited:
        return None
    expected = {s.key() for s in case.expected_sources}
    return sum(k in expected for k in cited) / len(cited)


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def summarize(results: list[CaseResult]) -> dict:
    answerable = [r for r in results if r.answerable_expected]
    unanswerable = [r for r in results if not r.answerable_expected]

    by_tag: dict[str, list[bool]] = defaultdict(list)
    by_confidence: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for tag in r.tags:
            by_tag[tag].append(r.correct)
        if r.confidence:
            by_confidence[r.confidence].append(r.correct)

    return {
        "cases": len(results),
        "errors": sum(r.error is not None for r in results),
        # Answers
        "accuracy": _rate([r.correct for r in results]),
        "answerable_accuracy": _rate([r.correct for r in answerable]),
        "refusal_accuracy": _rate([r.answerable_predicted is False for r in unanswerable]),
        "false_refusal_rate": _rate([r.answerable_predicted is False for r in answerable]),
        # Retrieval (answerable cases only)
        "retrieval_hit_rate": _rate([r.retrieval_hit for r in answerable if r.retrieval_hit is not None]),
        "retrieval_recall": _avg([r.retrieval_recall for r in answerable if r.retrieval_recall is not None]),
        # Citations
        "citation_precision": _avg([r.citation_precision for r in results if r.citation_precision is not None]),
        # Calibration: is "high" confidence actually right more often?
        "accuracy_by_confidence": {
            level: {"n": len(v), "accuracy": _rate(v)}
            for level, v in sorted(by_confidence.items())
        },
        "accuracy_by_tag": {
            tag: {"n": len(v), "accuracy": _rate(v)} for tag, v in sorted(by_tag.items())
        },
        "mean_latency_s": _avg([r.latency_s for r in results if r.latency_s is not None]),
    }