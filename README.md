# policy-guard-langchain-demo

This project demonstrates [Policy Guard](https://github.com/talalter/policy-guard) operating inside a real agentic pipeline - showing exactly where and how a safety layer plugs into a production RAG system. The integration point is a LangChain `BaseCallbackHandler` (`PolicyGuard`) that intercepts every final agent answer and posts it to Policy Guard's `POST /check` endpoint, which scores it against the full policy document and returns flagged spans with severity and confidence. The guard runs out-of-band from the agent's reasoning loop, so it is independently deployable and requires no changes to agent logic - just one HTTP call. Demo questions are deliberately adversarial (e.g. "How do I pull all user records directly from the database?") to show the guard catching real policy violations.

## Stack

**LangChain ReAct** + **GPT-4o** for the agent. Hybrid **BM25 + FAISS** retrieval for grounding answers in API docs. Policy Guard runs as a separate **FastAPI** service; this repo communicates with it over plain HTTP.

## Architecture

```
User question
      ↓
[ReAct agent loop]
  GPT-4o reasons about the question
      ↓ calls tool
[retrieve_docs tool]
  pipeline/retriever.py - BM25 + dense (FAISS) → RRF fusion → cross-encoder re-rank
  returns top-5 chunks
      ↓
  GPT-4o synthesises the final answer
      ↓
[PolicyGuard callback]
  calls POST /check on the running policy-guard service
  checks the final answer against the full policy document
  reports contradictions by severity, confidence, and explanation
```

**Key design decision - HTTP safety middleware:** `PolicyGuard` is a LangChain `BaseCallbackHandler` that calls `POST /check` on the policy-guard service after every chain completion. Decoupled and independently deployable - any agent can add faithfulness checking with one HTTP call, no changes to agent logic.

## Project layout

```
policy-guard-langchain-demo/
├── requirements.txt
├── .env.example
│
├── agent_demo.py               entry point - run 3 sample questions through the agent
├── ingest.py                   build the retrieval index from data/docs/
│
├── pipeline/                   core RAG components
│   ├── chunker.py              token-aware document chunking
│   ├── retriever.py            hybrid BM25 + FAISS retriever with cross-encoder re-ranking
│   └── ingest_config.py        named chunking strategies (baseline / large / small)
│
├── scripts/
│   └── test_retrieval.py       inspect retriever output for any ad-hoc query
│
└── data/
    ├── docs/                   Fynlo Imagen API documentation (20 plain-text files)
    ├── policy.txt              full policy document loaded on every safety check
    └── indices/baseline/       generated FAISS + BM25 index (not committed - build with python ingest.py)
```

## Setup

```bash
# 1. create and activate virtualenv
python -m venv venv && source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure secrets
cp .env.example .env
# set OPENAI_API_KEY in .env

# 4. build the retrieval index (one-time, ~2 min)
python ingest.py

# 5. run the demo
python agent_demo.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | - | Required |
| `GPT_MODEL` | `gpt-4o` | LangChain ChatOpenAI model |

## External dependency: policy-guard

The `PolicyGuard` class (`agent_demo.py`) is a LangChain `BaseCallbackHandler` that validates agent responses at runtime. On every chain completion it:

1. Loads `data/policy.txt` in full (not retrieved chunks - retrieval misses are silent failures in safety checks)
2. POSTs `{ context: policy, response: agent_answer }` to `POST /check`
3. Logs every violation with its severity, confidence score, and explanation

The integration is **decoupled** - no policy-guard code is imported. Any agent or service can add the same safety check with a single HTTP call.

**Start the service before running the demo:**

```bash
# In a separate terminal, from the policy-guard repo:
uvicorn backend.main:app --reload --port 8000
```

The `POLICY_GUARD_URL` environment variable controls where requests are sent (default: `http://localhost:8000`).
