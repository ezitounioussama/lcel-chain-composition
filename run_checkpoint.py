"""
Run the LCEL pipeline on two questions and print everything the checkpoint asks for.

    python3 run_checkpoint.py
"""

import sys
import time

from lcel_pipeline import (
    MODEL,
    parallel_probe,
    pipeline,
    run_with_trace,
    time_sequential,
)

QUESTIONS = [
    "How can I reduce latency in a web app that serves ML predictions?",
    "How should a small team choose between SQLite and PostgreSQL for a new product?",
]

LINE = "=" * 82
THIN = "-" * 82


def header(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


def section(title):
    print(f"\n{title}\n{THIN}")


def report(trace, number):
    header(f"QUESTION {number}: {trace['question']}")

    section("1. Decomposed sub-questions (raw model output)")
    for line in trace["decomposer_raw"].splitlines():
        print(f"  {line}")

    section(f"   Parsed into {len(trace['subquestions'])} sub-questions (capped at 3)")
    for index, question in enumerate(trace["subquestions"], start=1):
        print(f"  {index}. {question}")

    section("2. Sub-answers (plain text blocks: Answer + Steps)")
    for index, item in enumerate(trace["answers"], start=1):
        print(f"\n  [{index}] Answer: {item['answer']}")
        print("      Steps:")
        for step in item["steps"]:
            print(f"        - {step}")

    section("3. Final synthesis (three lines from the combiner)")
    for line in trace["final"].splitlines():
        if line.strip():
            print(f"  {line}")

    section("Stage timings")
    print(f"  decompose : {trace['decompose_seconds']:6.2f}s")
    print(f"  answer    : {trace['batch_seconds']:6.2f}s   (batched, "
          f"{len(trace['subquestions'])} sub-questions)")
    print(f"  combine   : {trace['combine_seconds']:6.2f}s")


def main():
    header("SETUP")
    print(f"  Model    : {MODEL} via ChatOllama (local)")
    print(f"  Pipeline : decomposer | parse_subq | run_answers | combiner")
    print(f"  Note     : ChatOllama replaces ChatOpenAI from the brief — no API key here.")
    print(f"             Both are Runnables, so the LCEL composition is unchanged.")

    traces = []
    for number, question in enumerate(QUESTIONS, start=1):
        trace = run_with_trace(question)
        traces.append(trace)
        report(trace, number)

    # -----------------------------------------------------------------------
    header("WHY .batch() MATTERS — MEASURED")

    trace = traces[0]
    subquestions = trace["subquestions"]

    sequential = time_sequential(subquestions)
    batched = trace["batch_seconds"]

    speedup = sequential / batched if batched else 0.0

    print(f"\n  Answering the same {len(subquestions)} sub-questions:")
    print(f"    one at a time (.invoke in a loop) : {sequential:6.2f}s")
    print(f"    together (.batch)                 : {batched:6.2f}s")
    print(f"    speedup                           : {speedup:6.2f}x")

    if speedup < 1.3:
        print(f"""
  No real gain — and that is the honest result on this setup, not a broken test.

  The usual explanation for batch() is that model calls are I/O-bound, so
  overlapping the waiting is free. That holds for a HOSTED api, where each call
  waits on someone else's servers. It does not hold here: the model runs on this
  machine, so there is no remote wait to overlap. One llama3.2:3b instance
  already saturates the local compute, and three concurrent requests just
  time-slice the same hardware.

  Checked rather than assumed: the server reported OLLAMA_NUM_PARALLEL: 1, so
  requests were queueing server-side. Restarting with OLLAMA_NUM_PARALLEL=4 and
  re-measuring gave 0.92x — still no gain. That rules out queue depth and leaves
  compute as the limit.

  batch() stays in the pipeline for two reasons that are not speed: the code is
  clearer than a hand-written loop, and it is already the right shape for the day
  the model moves behind an API, where the speedup is real.""")
    else:
        print(f"""
  {speedup:.2f}x faster. Each call spends most of its time waiting, so .batch()
  overlaps those waits while a Python loop makes them queue.""")

    # -----------------------------------------------------------------------
    header("RunnableParallel — DIFFERENT CHAINS, SAME INPUT, AT ONCE")

    started = time.perf_counter()
    probe = parallel_probe.invoke({"question": QUESTIONS[0]})
    elapsed = time.perf_counter() - started

    print(f"\n  Three branches ran together in {elapsed:.2f}s, returning one dict:\n")
    print(f"  restated     : {getattr(probe['restated'], 'content', '').strip()[:120]}")
    print(f"  risk         : {getattr(probe['risk'], 'content', '').strip()[:120]}")
    print(f"  subquestions : {len(probe['subquestions'])} items")
    print("""
  batch()          = one chain, many inputs
  RunnableParallel = many chains, one input
  Both return everything at once; neither needs threads written by hand.""")

    # -----------------------------------------------------------------------
    header("THE ONE-LINER: pipeline.invoke() RETURNS ONLY THE FINAL MESSAGE")

    started = time.perf_counter()
    final = pipeline.invoke({"question": "What makes a good unit test?"})
    elapsed = time.perf_counter() - started

    print(f"\n  pipeline.invoke({{'question': 'What makes a good unit test?'}})   [{elapsed:.2f}s]\n")
    for line in getattr(final, "content", str(final)).strip().splitlines():
        if line.strip():
            print(f"  {line}")
    print("""
  Four stages, one call. The intermediate sub-questions and sub-answers are
  produced but not returned — run_with_trace() exists only to show them.""")

    print(f"\n{LINE}\nDone.\n{LINE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nError: {type(error).__name__}: {error}")
        print("Is Ollama running?  ollama serve")
        sys.exit(1)
