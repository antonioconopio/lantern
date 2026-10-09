"""Tests for the eval harness itself: dataset integrity, scoring, and an
end-to-end smoke run with fake models (no API calls)."""

import asyncio
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from evals.dataset import CORPUS_DIR, EvalCase, load_dataset
from evals.judge import JudgeVerdict, judge_prompt
from evals.metrics import (
    CaseResult,
    citation_precision,
    retrieval_hit,
    retrieval_recall,
    summarize,
)
from evals.run_evals import build_eval_store, evaluate_case, run_cases
from lantern.answer import GroundedAnswer
from lantern.chains import build_rag_chain
from lantern.ingest import load_file
from lantern.retrieval import build_retriever
from tests.conftest import make_fake_llm

# ---- Dataset integrity --------------------------------------------------------


def test_dataset_loads_with_unique_ids():
    cases = load_dataset()
    assert len(cases) >= 20


def test_every_expected_source_exists_in_corpus():
    """Catches typos in filenames and pages that don't exist."""
    pages = {}
    for path in CORPUS_DIR.iterdir():
        docs = load_file(path, path.name, "check")
        pages[path.name] = {d.metadata["page"] for d in docs}

    for case in load_dataset():
        for src in case.expected_sources:
            assert src.document in pages, f"{case.id}: unknown document {src.document}"
            assert src.page in pages[src.document], f"{case.id}: no page {src.page}"


def test_dataset_covers_each_question_type():
    tags = {t for c in load_dataset() for t in c.tags}
    assert {"single", "multi", "unanswerable", "follow-up"} <= tags


def test_answerable_case_requires_sources():
    with pytest.raises(ValidationError):
        EvalCase(id="x", question="q", expected_answer="a", answerable=True)


def test_history_becomes_alternating_messages():
    case = EvalCase(
        id="x",
        question="And then?",
        expected_answer="a",
        answerable=False,
        history=[{"question": "q1", "answer": "a1"}],
    )
    assert [(m.type, m.content) for m in case.history_messages()] == [
        ("human", "q1"),
        ("ai", "a1"),
    ]


# ---- Metrics -------------------------------------------------------------------

DOC_A, DOC_B = ("a.pdf", 1), ("b.md", 1)


def case(answerable=True, sources=(DOC_A,)):
    return EvalCase(
        id="c",
        question="q",
        expected_answer="a",
        answerable=answerable,
        expected_sources=[{"document": d, "page": p} for d, p in sources] if answerable else [],
    )


def test_retrieval_hit_and_recall():
    multi = case(sources=(DOC_A, DOC_B))
    assert retrieval_hit(multi, [DOC_A, ("x", 1)]) is True
    assert retrieval_recall(multi, [DOC_A, ("x", 1)]) == 0.5
    assert retrieval_hit(multi, [("x", 1)]) is False


def test_retrieval_not_applicable_when_unanswerable():
    assert retrieval_hit(case(answerable=False), [DOC_A]) is None
    assert retrieval_recall(case(answerable=False), [DOC_A]) is None


def test_citation_precision():
    assert citation_precision(case(), [DOC_A, DOC_B]) == 0.5
    assert citation_precision(case(), []) is None
    assert citation_precision(case(answerable=False), [DOC_A]) == 0.0


def test_summarize():
    results = [
        CaseResult(id="1", tags=["single"], question="q", expected_answer="a",
                   answerable_expected=True, answerable_predicted=True,
                   confidence="high", correct=True, retrieval_hit=True, retrieval_recall=1.0),
        CaseResult(id="2", tags=["single"], question="q", expected_answer="a",
                   answerable_expected=True, answerable_predicted=False,
                   confidence="low", correct=False, retrieval_hit=False, retrieval_recall=0.0),
        CaseResult(id="3", tags=["unanswerable"], question="q", expected_answer="a",
                   answerable_expected=False, answerable_predicted=False,
                   confidence="low", correct=True),
    ]
    s = summarize(results)
    assert s["accuracy"] == 0.667
    assert s["answerable_accuracy"] == 0.5
    assert s["refusal_accuracy"] == 1.0
    assert s["false_refusal_rate"] == 0.5
    assert s["retrieval_hit_rate"] == 0.5
    assert s["accuracy_by_confidence"]["high"] == {"n": 1, "accuracy": 1.0}
    assert s["accuracy_by_tag"]["unanswerable"] == {"n": 1, "accuracy": 1.0}


# ---- evaluate_case with stub chain and judge -------------------------------------


class Doc:
    def __init__(self, filename, page):
        self.metadata = {"filename": filename, "page": page}


def stub_chain(answerable=True, citations=(1,), docs=(DOC_A,)):
    def run(_):
        return {
            "standalone_question": "q",
            "docs": [Doc(*d) for d in docs],
            "answer": GroundedAnswer(
                answer="ans", citations=list(citations), answerable=answerable, confidence="high"
            ),
        }
    return RunnableLambda(run)


def stub_judge(correct=True):
    return RunnableLambda(lambda _: JudgeVerdict(reason="r", correct=correct))


def run(coro):
    return asyncio.run(coro)


def test_correct_answer_is_judged_and_scored():
    r = run(evaluate_case(case(), stub_chain(), stub_judge(True)))
    assert r.correct and r.retrieval_hit and r.citation_precision == 1.0
    assert r.cited == [DOC_A]


def test_judge_can_mark_wrong():
    assert not run(evaluate_case(case(), stub_chain(), stub_judge(False))).correct


def test_unanswerable_correctly_declined_skips_judge():
    def exploding_judge(_):
        raise AssertionError("judge should not run")

    r = run(evaluate_case(case(answerable=False), stub_chain(answerable=False, citations=()),
                          RunnableLambda(exploding_judge)))
    assert r.correct and r.error is None


def test_answering_an_unanswerable_question_is_wrong():
    r = run(evaluate_case(case(answerable=False), stub_chain(answerable=True), stub_judge()))
    assert not r.correct


def test_false_refusal_is_wrong():
    r = run(evaluate_case(case(), stub_chain(answerable=False, citations=()), stub_judge()))
    assert not r.correct


def test_invented_citation_numbers_are_ignored():
    r = run(evaluate_case(case(), stub_chain(citations=(1, 9)), stub_judge()))
    assert r.cited == [DOC_A]


def test_errors_are_recorded_not_raised():
    def boom(_):
        raise RuntimeError("rate limited")

    r = run(evaluate_case(case(), RunnableLambda(boom), stub_judge()))
    assert r.error == "RuntimeError: rate limited" and not r.correct


def test_judge_prompt_includes_all_three_texts():
    text = judge_prompt.format(question="Q", expected_answer="REF", answer="CAND")
    assert "Q" in text and "REF" in text and "CAND" in text


# ---- End to end with fake models --------------------------------------------------


def test_smoke_run_over_real_corpus_and_dataset():
    """Ingests the real corpus and runs real dataset cases through the real
    chain; only the embeddings, LLM and judge are fakes."""
    store = build_eval_store(DeterministicFakeEmbedding(size=64), CORPUS_DIR, 1000, 150)
    chain = build_rag_chain(make_fake_llm(), build_retriever(store, k=4))
    cases = load_dataset()[:3] + [c for c in load_dataset() if "follow-up" in c.tags][:1]

    results = run(run_cases(cases, chain, stub_judge(True), concurrency=2))

    assert [r.error for r in results] == [None] * len(cases)
    assert all(len(r.retrieved) == 4 for r in results)
    assert summarize(results)["cases"] == len(cases)