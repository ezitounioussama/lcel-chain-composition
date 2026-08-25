# Results — LCEL Pipeline on Two Questions

Captured from a real run. Model `llama3.2:3b` via `ChatOllama`, local Ollama.
Raw log: [`docs/output.txt`](docs/output.txt).

```bash
ollama serve
python3 run_checkpoint.py
```

Pipeline: `decomposer | parse_subq | run_answers | combiner`

**One substitution from the brief:** `ChatOllama` in place of `ChatOpenAI`, because there is no
OpenAI key on this machine. That is the point of LCEL — a chat model is a Runnable, so the
composition is byte-for-byte the same with a different model class.

---

# Question 1

> How can I reduce latency in a web app that serves ML predictions?

## 1. Decomposed sub-questions

```
1. What are the primary causes of latency in a web app serving ML predictions?
2. How can I optimize the ML model and its deployment for faster inference times?
3. What are some strategies for reducing network latency and improving data transfer between the client and server?
```

Parsed into 3, capped at 3.

## 2. Sub-answers (Answer + Steps)

```
[1] Answer: Insufficient server resources, network connectivity issues, and high traffic.
    Steps:
      - Insufficient server resources (e.g., CPU, memory, or storage) can lead to slow
        processing times and increased latency.
      - Network connectivity issues, such as slow internet speeds or packet loss, can also
        cause delays in data transmission.
      - High traffic volumes can overwhelm the server, resulting in slower response times.

[2] Answer: Prune and optimize model weights, use efficient inference frameworks like
            TensorFlow Lite or Core ML.
    Steps:
      - Prune the model to reduce its size and complexity.
      - Use model quantization to reduce the precision of model weights.
      - Optimize the model for deployment on specific hardware platforms.
      - Use efficient inference frameworks like TensorFlow Lite or Core ML.
      - Profile and benchmark the model to identify performance bottlenecks.

[3] Answer: Implementing techniques such as caching, content delivery networks (CDNs), and
            data compression can help reduce network latency and improve data transfer.
    Steps:
      - Use caching to store frequently accessed data closer to the client.
      - Utilize CDNs to distribute data across multiple servers and reduce latency.
      - Apply data compression algorithms to reduce the amount of data transferred.
```

## 3. Final synthesis (three lines)

```
Final Answer: Insufficient server resources, network connectivity issues, and high traffic can
cause delays in data transmission, leading to slow processing times and increased latency.
Key points: - Insufficient server resources can lead to slow processing times and increased
latency. - Network connectivity issues and high traffic volumes can also cause delays in data
transmission. - Optimizing model weights, using efficient inference frameworks, and implementing
caching, CDNs, and data compression can help reduce network latency and improve data transfer.
Confidence: Medium
```

Stage timings: decompose 7.28s · answer 21.19s (3 batched) · combine 18.78s

---

# Question 2

> How should a small team choose between SQLite and PostgreSQL for a new product?

## 1. Decomposed sub-questions

```
1. What are the key differences between SQLite and PostgreSQL that would influence the choice?
2. What are the performance and scalability requirements for the new product, and how do SQLite and PostgreSQL compare in these areas?
3. What are the development and maintenance costs associated with each database, and how do these factors impact the choice?
```

## 2. Sub-answers (Answer + Steps)

```
[1] Answer: SQLite and PostgreSQL differ in scalability, complexity, and licensing.
    Steps:
      - SQLite is a self-contained, serverless database ideal for small to medium-sized
        applications with limited data.
      - PostgreSQL is a more scalable, feature-rich database suitable for large-scale
        applications with complex data needs, requiring more administrative effort.

[2] Answer: High performance and scalability are crucial for the new product, requiring
            databases that can handle large amounts of data and high traffic.
    Steps:
      - **Assess current data volume and traffic**: Determine the expected data volume and
        traffic to identify performance and scalability requirements.
      - **Evaluate database capabilities**: Research SQLite and PostgreSQL's performance and
        scalability features, such as indexing, caching, and connection pooling.
      - **Compare database architectures**: Analyze the storage engines and query execution plans.
      - **Test and benchmark databases**: Perform thorough testing and benchmarking.
      - **Choose the best database**: Select the one that best meets the requirements.

[3] Answer: The development and maintenance costs of a database vary depending on the type,
            complexity, and scalability requirements.
    Steps:
      - Research the costs of each database type to determine which fits the project's needs.
      - Evaluate the development time and resources required for each database.
      - Assess the ongoing maintenance costs, including licensing, server and personnel costs.
      - Weigh the trade-offs between development and maintenance costs.
```

## 3. Final synthesis (three lines)

```
Final Answer: SQLite and PostgreSQL differ in scalability, complexity, and licensing, with
PostgreSQL being more suitable for large-scale applications and high performance requirements.
Key points:
- PostgreSQL is more scalable and feature-rich than SQLite, making it suitable for large-scale
  applications.
- SQLite is ideal for small to medium-sized applications with limited data.
- The choice depends on the specific needs of the project, including performance, scalability,
  and licensing requirements.
Confidence: Medium
```

---

# Why `.batch()` matters — and where it does not

The brief asks to emphasise why batching matters. Measured on this setup, it did not:

```
Answering the same 3 sub-questions:
  one at a time (.invoke in a loop) :  20.35s
  together (.batch)                 :  23.91s
  speedup                           :   0.85x
```

That is the honest result, not a broken test, and the mechanism is worth understanding.

The usual explanation is that model calls are **I/O-bound**, so overlapping the waiting is free.
That holds for a **hosted** API, where every call waits on someone else's servers and ten requests
can be in flight at once. It does not hold here: the model runs on **this** machine, so there is no
remote wait to overlap. One `llama3.2:3b` instance already saturates the local compute, and three
concurrent requests just time-slice the same hardware.

**Checked rather than assumed.** The server reported `OLLAMA_NUM_PARALLEL: 1`, so requests were
queueing server-side — a plausible culprit. Restarting with `OLLAMA_NUM_PARALLEL=4` and
re-measuring gave **0.92x**, still no gain. That rules out queue depth and leaves compute as the
limit.

`.batch()` stays in the pipeline for two reasons that are not speed: the code is clearer than a
hand-written loop, and it is already the right shape for the day the model moves behind an API,
where the speedup is real.

---

# `RunnableParallel` — a different kind of concurrency

```
Three branches ran together in 12.27s, returning one dict:

restated     : How can I make a web app that uses machine learning predictions load faster
               and respond more quickly to user input?
risk         : The biggest risk when reducing latency in a web app that serves ML predictions
               is that overly optimized models or algorithms...
subquestions : 3 items
```

```
batch()          = one chain,   many inputs
RunnableParallel = many chains, one input
```

Both hand back everything at once, and neither needs a thread written by hand.

---

# The whole thing as one call

```
pipeline.invoke({'question': 'What makes a good unit test?'})   [47.75s]

Final Answer: A good unit test is independent, fast, reliable, and focused on a specific piece
of functionality, ensuring code coverage and reliability by thoroughly testing individual units
of code.
Key points: - A unit test should be isolated from dependencies and external factors. - Mocking
and stubbing are used to control the flow of input/output in unit tests. - Unit tests should
cover all possible execution paths and be run regularly to detect defects.
Confidence: high
```

Four stages, one call. `invoke()` returns only the final message — `run_with_trace()` exists
purely to expose the intermediate sub-questions and sub-answers this report needs.

---

# Unit tests

```
$ python3 tests.py
Ran 17 tests in 3.871s

OK
```

The optional assignment test — *the decomposer returns at most 3 sub-questions* — is checked twice:

- **Against fake model output**, including a model that returns six items and four awkward
  formats (`1.`, `1)`, indented, blank-line separated). All capped at 3.
- **Against the live model**, asserting 1–3 sub-questions, all non-empty strings, with the
  `"1. "` numbering stripped — if numbering survived the parse it would end up inside the next
  prompt.

The cap is enforced in code (`subquestions[:3]`), not left to the prompt. The prompt says
"up to 3", but a prompt is a request, not a guarantee.

---

Author: **Oussama Ezitouni**

---

# The written answer (checkpoint deliverable, 150 words)

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
