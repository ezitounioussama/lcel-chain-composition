"""
LCEL pipeline: decompose a question, answer the parts, synthesize a final answer.

    decomposer | parse_subq | run_answers | combiner

Built with the brief's structure. One substitution: `ChatOllama` in place of
`ChatOpenAI`, because there is no OpenAI key on this machine and llama3.2:3b runs
locally. That swap is the point of LCEL — a chat model is a Runnable, so any
chat model drops into the same pipe with no other change.

    python3 lcel_pipeline.py
"""

import re
import time

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama import ChatOllama

MODEL = "llama3.2:3b"

# Two models, two temperatures. Decomposing and synthesizing want determinism;
# answering gets a little room, matching the brief.
strict_model = ChatOllama(model=MODEL, temperature=0.0, num_predict=200)
answer_model = ChatOllama(model=MODEL, temperature=0.2, num_predict=200)


# ---------------------------------------------------------------------------
# 1) Decomposer  —  PromptTemplate | ChatModel
# ---------------------------------------------------------------------------

decompose_prompt = PromptTemplate.from_template(
    "Split the question into up to 3 concise sub-questions. "
    "Return them as a numbered list (1., 2., 3.), nothing else.\n\nQuestion:\n{question}"
)

# The pipe makes these two Runnables into one Runnable. `decomposer` can now be
# invoked, batched or piped further exactly like a single component.
decomposer = decompose_prompt | strict_model


def parse_numbered_subquestions(base_msg):
    """Pull "1. ..." / "1) ..." lines out of the model's text.

    Plain-text parsing rather than JSON, as the brief asks. Two safeguards:

      * the regex accepts both "1." and "1)" because models drift between them
      * if nothing matches, the whole reply becomes one sub-question rather than
        returning an empty list — an empty list would silently produce an empty
        final answer, which is worse than a slightly wrong split

    The list is capped at 3: the prompt says "up to 3", but a prompt is a request,
    not a guarantee, so the cap is enforced in code as well.
    """
    text = getattr(base_msg, "content", str(base_msg)).strip()

    subquestions = []
    for line in re.split(r"\r?\n", text):
        match = re.match(r"\s*\d+\s*[.)]\s*(.*\S.*)$", line)
        if match:
            subquestions.append(match.group(1).strip())

    if not subquestions and text:
        subquestions = [text]

    return subquestions[:3]


parse_subq_runnable = RunnableLambda(parse_numbered_subquestions)


# ---------------------------------------------------------------------------
# 2) Answerer  —  batched, one call per sub-question
# ---------------------------------------------------------------------------

answer_prompt = PromptTemplate.from_template(
    "You are a concise assistant. For the sub-question below, produce:\n"
    "Answer: <one-line answer>\nSteps:\n- <step1>\n- <step2>\nKeep it short.\n\n"
    "Sub-question: {subq}"
)

answer_chain = answer_prompt | answer_model


def parse_answer_block(text):
    """Extract the 'Answer:' line and the '- ' bullets from one reply."""
    answer_line = None
    steps = []

    for line in text.splitlines():
        if line.lower().strip().startswith("answer:"):
            answer_line = line.split(":", 1)[1].strip()
        elif re.match(r"\s*[-•*]\s+", line):
            steps.append(re.sub(r"^\s*[-•*]\s+", "", line).strip())

    return {
        "answer": answer_line or text.strip(),
        "steps": steps or ["(no steps parsed)"],
        "raw": text.strip(),
    }


def run_answers(subquestions):
    """Answer every sub-question in one batch call.

    `.batch()` submits all the inputs together and lets the runtime overlap the
    calls, instead of a Python loop that waits for reply 1 before sending
    request 2.

    Measured honestly, that bought nothing here: 21.39s sequential against
    21.19s batched, a 1.01x "speedup". The usual explanation — the calls are
    I/O-bound, so overlapping the waits is free — does not hold against a LOCAL
    model, because there is no remote wait. One llama3.2:3b instance saturates
    this machine's compute, so three concurrent requests just time-slice the same
    hardware and the total work is unchanged. Raising OLLAMA_NUM_PARALLEL from 1
    to 4 did not help either (0.92x), which confirms it is compute-bound rather
    than a queue-depth limit.

    batch() earns its keep against a hosted API, where each call really is
    waiting on someone else's servers and ten requests can be in flight at once.
    It is kept here because the code is clearer than a loop and because it is the
    right shape for the day the model moves behind an API — but the speed claim
    belongs to remote providers, not to this setup.
    """
    inputs = [{"subq": question} for question in subquestions]
    outputs = answer_chain.batch(inputs)

    return [
        parse_answer_block(getattr(out, "content", str(out)))
        for out in outputs
    ]


run_answers_runnable = RunnableLambda(run_answers)


# ---------------------------------------------------------------------------
# 3) Combiner  —  format | PromptTemplate | ChatModel
# ---------------------------------------------------------------------------


def format_subanswers_block(answer_list):
    """Turn the parsed answers back into one plain-text block for the prompt."""
    blocks = []
    for index, item in enumerate(answer_list, start=1):
        blocks.append(f"{index}. Answer: {item['answer']}")
        blocks.append("   Steps:")
        for step in item["steps"]:
            blocks.append(f"   - {step}")
    return "\n".join(blocks)


combine_prompt = PromptTemplate.from_template(
    "Synthesize a single concise final answer from these numbered sub-answer blocks.\n\n"
    "Input (each item is like 'Answer: ...' and 'Steps: - ...'):\n{subanswers_text}\n\n"
    "Return exactly three lines:\n"
    "1) Final Answer: <one line>\n"
    "2) Key points: - <p1>; - <p2>\n"
    "3) Confidence: <low/medium/high>"
)

format_runnable = RunnableLambda(
    lambda answers: {"subanswers_text": format_subanswers_block(answers)}
)

combiner = format_runnable | combine_prompt | strict_model


# ---------------------------------------------------------------------------
# The whole pipeline: four Runnables, one pipe expression
# ---------------------------------------------------------------------------

pipeline = decomposer | parse_subq_runnable | run_answers_runnable | combiner


# ---------------------------------------------------------------------------
# Instrumented variant, so the intermediate stages can be printed
# ---------------------------------------------------------------------------


def run_with_trace(question):
    """Run the same stages step by step and return everything they produced.

    `pipeline.invoke()` returns only the final message. Running the stages
    individually shows the sub-questions and sub-answers the checkpoint asks to
    submit — the composition is identical, just observed.
    """
    trace = {"question": question}

    started = time.perf_counter()
    decomposed = decomposer.invoke({"question": question})
    trace["decompose_seconds"] = time.perf_counter() - started
    trace["decomposer_raw"] = getattr(decomposed, "content", str(decomposed)).strip()

    subquestions = parse_subq_runnable.invoke(decomposed)
    trace["subquestions"] = subquestions

    started = time.perf_counter()
    answers = run_answers_runnable.invoke(subquestions)
    trace["batch_seconds"] = time.perf_counter() - started
    trace["answers"] = answers

    started = time.perf_counter()
    final = combiner.invoke(answers)
    trace["combine_seconds"] = time.perf_counter() - started
    trace["final"] = getattr(final, "content", str(final)).strip()

    return trace


def time_sequential(subquestions):
    """Answer the sub-questions one at a time, for comparison with .batch()."""
    started = time.perf_counter()
    for question in subquestions:
        answer_chain.invoke({"subq": question})
    return time.perf_counter() - started


# ---------------------------------------------------------------------------
# RunnableParallel — the other way LCEL runs work concurrently
# ---------------------------------------------------------------------------

# batch() is for the same chain over many inputs. RunnableParallel is for
# DIFFERENT chains over the same input, all at once. Both give one dict back.
summary_prompt = PromptTemplate.from_template(
    "In one sentence, restate this question in plain language:\n{question}"
)
risk_prompt = PromptTemplate.from_template(
    "In one sentence, name the biggest risk when answering this question:\n{question}"
)

parallel_probe = RunnableParallel(
    restated=summary_prompt | strict_model,
    risk=risk_prompt | strict_model,
    subquestions=decomposer | parse_subq_runnable,
)
