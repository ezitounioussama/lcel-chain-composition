# LangChain LCEL — Chain Composition

Take a question, break it into sub-questions, answer each one, then synthesize a final answer —
built as four stages piped into a single Runnable and invoked with one call.

One thing worth recording: `.batch()` is supposed to be the easy speed-up, and against a local
model it measured **0.85x** — slower than looping. There is no remote wait to overlap when the
model is on your own machine, and one instance already saturates it; raising
`OLLAMA_NUM_PARALLEL` to 4 changed nothing, which confirms compute is the limit rather than
concurrency. Against a hosted API the same call would be a real win.

```bash
ollama serve && ollama pull llama3.2:3b
python3 -m venv .venv && .venv/bin/python -m pip install langchain-core langchain-ollama
.venv/bin/python run_checkpoint.py      # the checkpoint output
.venv/bin/python tests.py               # 17 tests
```

`ChatOllama` replaces `ChatOpenAI` from the brief (no API key here). Both are Runnables, so the
composition is unchanged — which is the whole argument for LCEL.

## Also in this repo

- **[RESULTS.md](RESULTS.md)** — both questions end to end, the batch measurements, and the
  written 150-word answer the checkpoint asks for
- [`docs/output.txt`](docs/output.txt) — raw log

---

Author: **Oussama Ezitouni**
