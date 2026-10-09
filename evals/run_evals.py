"""Run the eval set against the real pipeline and save a scored report.

    python -m evals.run_evals                     # defaults from your .env
    python -m evals.run_evals --k 2 --chunk-size 400
    python -m evals.run_evals --only nw-century --only ab-e07
    python -m evals.run_evals --fail-under 0.8    # exit code 1 if accuracy is lower (for CI)

The corpus is ingested into a fresh in-memory collection on every run, so
evals never touch your real data, and chunking settings take effect each time.
Each run is saved to evals/results/ with its settings, so runs can be compared.
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from evals.dataset import CORPUS_DIR, EvalCase, load_dataset
from evals.judge import build_judge
from evals.metrics import (
    CaseResult,
    citation_precision,
    retrieval_hit,
    retrieval_recall,
    summarize,
)
from lantern.chains import build_rag_chain
from lantern.ingest import SUPPORTED_EXTENSIONS, ingest_file
from lantern.retrieval import build_retriever

RESULTS_DIR = Path(__file__).parent / "results"


def build_eval_store(
    embeddings: Embeddings, corpus_dir: Path, chunk_size: int, chunk_overlap: int
) -> Chroma:
    store = Chroma(
        collection_name=f"lantern-eval-{uuid.uuid4().hex}",
        embedding_function=embeddings,
    )
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    for path in files:
        ingest_file(path, path.name, store, chunk_size, chunk_overlap)
    return store


def _key(doc) -> tuple[str, int]:
    return (doc.metadata["filename"], doc.metadata["page"])


async def evaluate_case(case: EvalCase, chain: Runnable, judge: Runnable) -> CaseResult:
    result = CaseResult(
        id=case.id,
        tags=case.tags,
        question=case.question,
        expected_answer=case.expected_answer,
        answerable_expected=case.answerable,
    )
    try:
        start = time.perf_counter()
        out = await chain.ainvoke(
            {"question": case.question, "history": case.history_messages()}
        )
        result.latency_s = round(time.perf_counter() - start, 2)

        answer, docs = out["answer"], out["docs"]
        result.standalone_question = out["standalone_question"]
        result.answer = answer.answer
        result.answerable_predicted = answer.answerable
        result.confidence = answer.confidence
        result.retrieved = [_key(d) for d in docs]
        result.cited = [_key(docs[n - 1]) for n in sorted(set(answer.citations)) if 1 <= n <= len(docs)]

        result.retrieval_hit = retrieval_hit(case, result.retrieved)
        result.retrieval_recall = retrieval_recall(case, result.retrieved)
        result.citation_precision = citation_precision(case, result.cited)

        if not case.answerable:
            # Correct means declining to answer; no judge needed.
            result.correct = answer.answerable is False
            result.judge_reason = (
                "Correctly declined." if result.correct else "Answered a question the documents don't cover."
            )
        elif not answer.answerable:
            result.correct = False
            result.judge_reason = "Said it couldn't find an answer the documents contain."
        else:
            verdict = await judge.ainvoke(
                {
                    "question": case.question,
                    "expected_answer": case.expected_answer,
                    "answer": answer.answer,
                }
            )
            result.correct = verdict.correct
            result.judge_reason = verdict.reason
    except Exception as e:  # one failing case shouldn't stop the run
        result.error = f"{type(e).__name__}: {e}"
    return result


async def run_cases(
    cases: list[EvalCase], chain: Runnable, judge: Runnable, concurrency: int = 4
) -> list[CaseResult]:
    """Run cases concurrently (bounded, to stay under API rate limits)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(case):
        async with semaphore:
            r = await evaluate_case(case, chain, judge)
            mark = "ERR " if r.error else ("PASS" if r.correct else "FAIL")
            print(f"  {mark}  {case.id}", flush=True)
            return r

    return await asyncio.gather(*(bounded(c) for c in cases))


def _pct(value) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def print_report(summary: dict, results: list[CaseResult]) -> None:
    print("\n=== Summary ===")
    rows = [
        ("Accuracy (all cases)", _pct(summary["accuracy"])),
        ("  answerable questions", _pct(summary["answerable_accuracy"])),
        ("  correctly declined unanswerable", _pct(summary["refusal_accuracy"])),
        ("  false refusals", _pct(summary["false_refusal_rate"])),
        ("Retrieval hit rate", _pct(summary["retrieval_hit_rate"])),
        ("Retrieval recall", _pct(summary["retrieval_recall"])),
        ("Citation precision", _pct(summary["citation_precision"])),
    ]
    for label, value in rows:
        print(f"{label:<36}{value}")
    print(f"{'Mean latency':<36}{summary['mean_latency_s']}s")
    if summary["errors"]:
        print(f"{'Errors':<36}{summary['errors']}")

    print("\nBy tag:")
    for tag, s in summary["accuracy_by_tag"].items():
        print(f"  {tag:<20}{_pct(s['accuracy'])}  (n={s['n']})")
    print("By model confidence:")
    for level, s in summary["accuracy_by_confidence"].items():
        print(f"  {level:<20}{_pct(s['accuracy'])}  (n={s['n']})")

    failed = [r for r in results if not r.correct]
    if failed:
        print("\n=== Failed cases ===")
        for r in failed:
            print(f"\n[{r.id}] {r.question}")
            if r.standalone_question and r.standalone_question != r.question:
                print(f"  rewritten:  {r.standalone_question}")
            if r.error:
                print(f"  error:      {r.error}")
                continue
            print(f"  expected:   {r.expected_answer}")
            print(f"  got:        {r.answer}")
            print(f"  why:        {r.judge_reason}")
            if r.retrieval_hit is False:
                print("  retrieval:  MISSED the expected source (a retrieval problem, not the prompt)")


def parse_args(argv, settings) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Lantern's eval set.")
    p.add_argument("--k", type=int, default=settings.retrieval_k, help="chunks to retrieve")
    p.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    p.add_argument("--model", default=settings.llm_model, help="model being evaluated")
    p.add_argument("--judge-model", default=None, help="defaults to --model")
    p.add_argument("--only", action="append", help="run only these case ids (repeatable)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--fail-under", type=float, default=None, help="exit 1 if accuracy is below this")
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Imported here so the module can be imported (and tested) without a .env.
    from app.config import get_embeddings, get_settings

    settings = get_settings()
    args = parse_args(argv, settings)

    cases = load_dataset()
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
        if not cases:
            print("No cases matched --only.")
            return 2

    print(f"Ingesting corpus (chunk size {args.chunk_size}, overlap {args.chunk_overlap})...")
    store = build_eval_store(get_embeddings(), CORPUS_DIR, args.chunk_size, args.chunk_overlap)

    llm = init_chat_model(args.model)
    judge_llm = init_chat_model(args.judge_model) if args.judge_model else llm
    chain = build_rag_chain(llm, build_retriever(store, k=args.k))

    print(f"Running {len(cases)} cases with {args.model} (k={args.k})...")
    results = asyncio.run(run_cases(cases, chain, build_judge(judge_llm), args.concurrency))
    summary = summarize(results)
    print_report(summary, results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    config = {
        "model": args.model,
        "judge_model": args.judge_model or args.model,
        "embedding_model": settings.embedding_model,
        "k": args.k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    out.write_text(
        json.dumps(
            {"config": config, "summary": summary, "results": [r.model_dump() for r in results]},
            indent=2,
        )
    )
    print(f"\nSaved {out}")

    if args.fail_under is not None and (summary["accuracy"] or 0) < args.fail_under:
        print(f"Accuracy {summary['accuracy']} is below --fail-under {args.fail_under}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())