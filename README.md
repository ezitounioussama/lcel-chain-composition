# LangChain LCEL — Chain Composition

Decompose a question into sub-questions, answer them, synthesize a final answer — composed with
LCEL pipes and run on a local model.

> **[RESULTS.md](RESULTS.md)** — both questions, sub-questions, sub-answers, 3-line syntheses,
> and the measured batch comparison.

```bash
ollama serve && ollama pull llama3.2:3b
python3 -m venv .venv && .venv/bin/python -m pip install langchain-core langchain-ollama
.venv/bin/python run_checkpoint.py      # the checkpoint output
.venv/bin/python tests.py               # 17 tests
```

| File | Contents |
|---|---|
| `lcel_pipeline.py` | The chain: prompts, models, parsers, `RunnableParallel` |
| `run_checkpoint.py` | Runs it on two questions and prints every stage |
| `tests.py` | 17 tests, incl. the "at most 3 sub-questions" assertion |
| `docs/output.txt` | Raw log |

`ChatOllama` replaces `ChatOpenAI` from the brief (no API key here). Both are Runnables, so the
composition is unchanged — which is the whole argument for LCEL.

## Where the pipes are, and why batch helps (150 words)

Three pipes build the chain. `decompose_prompt | strict_model` turns a template and a model into
one Runnable. `answer_prompt | answer_model` does the same for the answering step.
`format_runnable | combine_prompt | strict_model` chains a plain Python function into a prompt and
a model. The four stages then compose into one object:
`decomposer | parse_subq | run_answers | combiner`, invoked with a single call.

`RunnableLambda` is what lets ordinary functions — regex parsers — sit in the pipe as equals.

`.batch()` sends all sub-questions at once instead of looping. Against a hosted API that is a real
win: each call is waiting on someone else's servers, so the waits overlap. Measured against a
**local** model it gave 0.85x — no remote wait exists, and one model instance already saturates
this machine. Raising `OLLAMA_NUM_PARALLEL` to 4 changed nothing, confirming compute is the limit.

---

Author: **Oussama Ezitouni**
