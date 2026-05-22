# CLAUDE.md - Agent Policy Guard Demo

This file tells Claude Code everything it needs to know about this project.
Read this file completely before touching any code.

---

## Standing instructions for every response

The user is building this project to land a job as software engineer or AI engineer. Every answer must factor in:
1. **Impressiveness** - would a senior engineer find this decision noteworthy?
2. **Best practice** - always recommend the industry-standard approach.

If there is a "quick" answer and an "impressive + correct" answer, always give the latter.

---

## What this project is

A demo that shows the **policy-guard** (policy guard) working inside a real
agentic pipeline.

The main repository is the policy-guard. This repo is its demo client:
a LangChain ReAct agent that retrieves Fynlo Imagen API docs, answers user questions,
then validates its own answer against a policy document by calling POST /check.

If the agent's answer advises something that violates the policy (e.g. "query the
database directly" when the policy says that is forbidden), the policy guard catches it.

**This project exists to answer one interview question:**
> "How does the policy guard work in a real system?"

The answer is: drop in a callback, load the policy doc, call POST /check. Done.

**Domain:** Fynlo Imagen - a fictional API company. Docs cover standard API topics.
`data/policy.txt` contains explicit forbidden-operation rules used by the policy guard.

---

## How we work together

- Work on one component at a time.
- Before writing code, explain the concept.
- After writing code, give the exact command to test it and the expected output.
- Keep functions small and single-purpose. No function longer than 30 lines.
- Every function gets a docstring. Every module gets a module-level docstring.
- Never use print() - use Python's logging module.

---

## External dependency: policy guard service

This project calls the policy-guard over HTTP - it does not import its code.

**Service URL:** `http://localhost:8000` (dev) or `POLICY_GUARD_URL` env var


### POST /check

```json
{
  "context": "<full policy document text>",
  "response": "<agent answer or planned action>"
}
```

`context` is always the **full `data/policy.txt`** - not retrieved chunks.
Loading the full policy doc guarantees every rule is checked regardless of what
the retriever returned. Retrieval misses are silent failures in safety checks.

### Response schema

```python
class ContradictionReport(BaseModel):
    faithfulness_score: float          # 0-1, higher = more compliant
    contradictions: list[Contradiction]
    method_used: DetectionMethod
    processing_time_ms: float

class Contradiction(BaseModel):
    response_span: str   # phrase in the agent answer that violates policy
    context_span: str    # the policy rule being violated
    explanation: str
    severity: Severity   # "direct" | "partial" | "multi-hop"
    confidence: float
```

### Compliance tiers

```
score >= 0.85   →  PASS   compliant
score  0.5-0.85 →  WARN   possible violation - show report
score < 0.50    →  BLOCK  clear violation - warn user
```


---

## Architecture

```
policy-guard-langchain-demo/
├── CLAUDE.md
├── requirements.txt
├── .env.example
│
├── pipeline/
│   ├── chunker.py       # token-aware chunking for the retriever index
│   └── retriever.py     # hybrid BM25 + dense + cross-encoder re-ranking
│
├── data/
│   ├── policy.txt       # full Fynlo Imagen access & security policy (always loaded)
│   └── docs/            # API documentation (chunked and indexed for retrieval)
│
└── agent_demo.py        # LangChain ReAct agent + PolicyGuard callback
```

---

## Pipeline

```
User question
      ↓
[ReAct agent - agent_demo.py]
  retrieve_docs(query) → hybrid BM25 + dense + re-rank → top-5 API doc chunks
  GPT-4o generates answer from those chunks
      ↓
[PolicyGuard.on_chain_end - agent_demo.py]
  policy = load("data/policy.txt")      ← full doc, always
  response = outputs["output"]          ← agent's final answer
  POST /check → ContradictionReport
  PASS / WARN / BLOCK
```

---

## Files

### pipeline/chunker.py (Part 1)
Splits raw doc text into overlapping token-aware chunks for the retriever index.
The policy doc is NOT chunked - it is loaded whole.

### pipeline/retriever.py (Part 2)
Hybrid retrieval: BM25 (keyword) + FAISS dense (semantic) → RRF fusion → cross-encoder re-rank.
Used by the agent to answer technical questions about the API.

### agent_demo.py (Part 3)
The main demo file. Contains:
- `PolicyGuard(BaseCallbackHandler)` - loads policy doc, calls POST /check, logs tier
- `_build_retriever_tool(retriever)` - wraps retriever as a LangChain Tool
- `_build_executor(llm, tools, guard)` - ReAct AgentExecutor
- `_preflight(url)` - checks service is reachable
- `main()` - runs demo questions

Demo questions include policy-violation scenarios:
- "How do I pull all user records directly from the database?" → BLOCK
- "Can I skip authentication for internal calls?" → WARN/BLOCK
- "What HTTP status does rate limiting return?" → PASS

---

## Environment variables

```bash
OPENAI_API_KEY=sk-...
POLICY_GUARD_URL=http://localhost:8000
POLICY_DOC_PATH=data/policy.txt
GPT_MODEL=gpt-4o
RAG_TOP_K=5
```

---

## Key concepts for interviews

**Why full policy doc, not retrieved chunks**
Retrieval misses are silent failures in safety-critical checks. If the retriever doesn't
surface a relevant rule, the violation goes undetected. The full policy doc is always
loaded - completeness over latency is the right tradeoff for compliance enforcement.

**PolicyGuard as a side-channel observer**
It's a `BaseCallbackHandler`. It doesn't modify the agent's reasoning loop - it
observes the final output and validates it out-of-band. This is how production safety
systems work: decoupled from the agent, independently deployable, swappable.

**Hybrid retrieval**
BM25 catches exact keyword matches (method names, HTTP verbs, header names) that
dense retrieval misses. RRF merges both ranked lists without score normalization.
Cross-encoder re-ranking then selects the top-5 with full attention over each pair.
