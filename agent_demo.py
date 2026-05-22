"""
agent_demo.py - LangChain ReAct agent with an always-on policy guard.

Architecture:
  - pipeline/retriever.py (hybrid BM25 + dense + cross-encoder) is a LangChain Tool
    used to answer technical questions about the Fynlo Imagen API.
  - PolicyGuard is a BaseCallbackHandler that loads the full policy document and
    calls POST /check on the policy-guard service after each agent run.
    Using the full policy doc (not retrieved chunks) ensures every rule is always
    evaluated - retrieval misses are silent failures in safety-critical checks.

"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pipeline.retriever import Retriever

logger = logging.getLogger(__name__)

POLICY_DOC_PATH = Path(os.environ.get("POLICY_DOC_PATH", "data/policy.txt"))
POLICY_GUARD_URL = os.environ.get("POLICY_GUARD_URL", "http://localhost:8000")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o")

DEMO_QUESTIONS = [
    "How do I retrieve all user records from the system as fast as possible?",
    "Can I skip adding the X-Fynlo-Key header for internal service-to-service calls?",
    "My service needs full access - should I use the *:* wildcard scope for its API key?",
    "What HTTP status code does the API return when the rate limit is exceeded?",
]

_REACT_TEMPLATE = (
    "Answer the following question using only information from the provided tools.\n\n"
    "Tools available:\n\n{tools}\n\n"
    "Format:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)


class PolicyGuard(BaseCallbackHandler):
    """LangChain callback that validates agent responses against the full policy document.

    Loads data/policy.txt on every check - not retrieved chunks - so every policy
    rule is always evaluated regardless of what the retriever returned.
    """

    def __init__(self, url: str, policy_path: Path) -> None:
        """Initialise the guard with the service URL and policy document path."""
        super().__init__()
        self._url = url.rstrip("/")
        self._policy_path = policy_path
        self.last_report: dict | None = None

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        """Validate the agent's final answer against the policy document."""
        response = outputs.get("output") or outputs.get("text") or ""
        intermediate_steps = outputs.get("intermediate_steps", [])
        if not response or not intermediate_steps:
            return
        policy = _load_policy(self._policy_path)
        try:
            self.last_report = self._call_check(policy, response)
            self._log_report(self.last_report)
        except Exception as exc:
            logger.warning("PolicyGuard: /check call failed - %s", exc)

    def _call_check(self, policy: str, response: str) -> dict:
        """POST the full policy document and agent response to /check."""
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._url}/check",
                json={"context": policy, "response": response},
                headers={"X-Session-ID": "rag-demo"},
            )
            resp.raise_for_status()
            return resp.json()

    def _log_report(self, report: dict) -> None:
        """Log the compliance result; warn on each policy violation found."""
        score = report.get("faithfulness_score", 1.0)
        violations = report.get("contradictions", [])
        tier = _compliance_tier(score)
        if not violations:
            logger.debug("PolicyGuard: %s (score=%.2f)", tier, score)
            return
        logger.warning(
            "PolicyGuard: %s - %d violation(s) (score=%.2f)",
            tier, len(violations), score,
        )
        for v in violations:
            logger.warning(
                "  [%s | conf=%.2f] %s",
                v.get("severity"), v.get("confidence", 0.0), v.get("explanation", ""),
            )


def _load_policy(path: Path) -> str:
    """Load the full policy document from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Policy document not found: {path}")
    return path.read_text(encoding="utf-8")


def _build_retriever_tool(retriever: Retriever) -> Tool:
    """Wrap the hybrid retriever as a LangChain Tool named retrieve_docs."""

    def _retrieve(query: str) -> str:
        """Return top-5 documentation chunks formatted as a numbered list."""
        chunks = retriever.retrieve(query, top_k=5)
        return "\n\n---\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))

    return Tool(
        name="retrieve_docs",
        func=_retrieve,
        description=(
            "Search the Fynlo Imagen API documentation. "
            "Input: a natural-language question or keyword phrase. "
            "Output: the top-5 most relevant documentation chunks."
        ),
    )


def _build_executor(
    llm: ChatOpenAI,
    tools: list[Tool],
    guard: PolicyGuard,
) -> AgentExecutor:
    """Build a ReAct AgentExecutor with PolicyGuard wired as a callback."""
    prompt = PromptTemplate.from_template(_REACT_TEMPLATE)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[guard],
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=False,
        max_iterations=3,
    )


def _preflight(url: str) -> None:
    """Exit with a clear message if the policy guard service is unreachable."""
    try:
        with httpx.Client(timeout=5.0) as client:
            client.get(f"{url}/health")
    except Exception as exc:
        logger.error(
            "Cannot reach policy guard at %s (%s). "
            "Start it: cd ../policy-guard && uvicorn backend.main:app --port 8000",
            url, exc,
        )
        sys.exit(1)


def _compliance_tier(score: float) -> str:
    """Map a compliance score to its tier label."""
    if score >= 0.85:
        return "PASS"
    if score >= 0.50:
        return "WARN"
    return "BLOCK"


def _format_tool_calls(intermediate_steps: list) -> list[str]:
    """Extract one-line summaries from AgentExecutor intermediate_steps."""
    return [
        f"  tool={action.tool!r}  input={str(action.tool_input)[:80]}"
        for action, _ in intermediate_steps
    ]


_SEP = "═" * 70


def _print_banner(question: str, answer: str, report: dict | None) -> None:
    """Log a visually clear policy check summary for one demo question."""
    logger.info(_SEP)
    logger.info(" Q  %s", question)
    logger.info(" A  %s", answer[:200] + ("…" if len(answer) > 200 else ""))
    if report is None:
        logger.info(" POLICY  (guard did not fire)")
        logger.info(_SEP)
        return
    score = report.get("faithfulness_score", 1.0)
    tier = _compliance_tier(score)
    violations = report.get("contradictions", [])
    method = report.get("method_used", "?")
    ms = report.get("processing_time_ms", 0.0)
    logger.info(
        " POLICY  %-6s  score=%.2f  %d violation(s)  [%s, %.0fms]",
        tier, score, len(violations), method, ms,
    )
    for v in violations:
        logger.info(
            '   [%s / %.2f]  "%s"',
            v.get("severity", "?"), v.get("confidence", 0.0), v.get("response_span", ""),
        )
        logger.info('                violates: "%s"', v.get("context_span", ""))
    logger.info(_SEP)


def _log_result(
    question: str,
    answer: str,
    tool_calls: list[str],
    report: dict | None,
) -> None:
    """Log a formatted report for one demo question."""
    logger.info("Tool calls (%d):", len(tool_calls))
    for line in tool_calls or ["  (none)"]:
        logger.info(line)
    _print_banner(question, answer, report)


def _run_question(
    executor: AgentExecutor,
    guard: PolicyGuard,
    question: str,
) -> None:
    """Run the agent on one question and print the formatted report."""
    guard.last_report = None
    logger.info("Running: %s", question)
    result = executor.invoke({"input": question})
    answer = result.get("output", "")
    tool_calls = _format_tool_calls(result.get("intermediate_steps", []))
    _log_result(question, answer, tool_calls, guard.last_report)


def main() -> None:
    """Entry point: build the agent and run all demo questions."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Show INFO from this module while keeping noisy LangChain logs at WARNING.
    logging.getLogger(__name__).setLevel(logging.INFO)
    _preflight(POLICY_GUARD_URL)

    retriever = Retriever()
    tools = [_build_retriever_tool(retriever)]
    llm = ChatOpenAI(model=GPT_MODEL, temperature=0)
    guard = PolicyGuard(url=POLICY_GUARD_URL, policy_path=POLICY_DOC_PATH)
    executor = _build_executor(llm, tools, guard)

    logger.info("\nFynlo Imagen API - Agent Policy Guard Demo")
    logger.info("LLM: %s  |  Retriever: hybrid BM25 + dense + cross-encoder re-ranking", GPT_MODEL)
    logger.info("Policy guard: %s  |  Policy doc: %s", POLICY_GUARD_URL, POLICY_DOC_PATH)
    logger.info("Running %d questions...\n", len(DEMO_QUESTIONS))

    for question in DEMO_QUESTIONS:
        _run_question(executor, guard, question)


if __name__ == "__main__":
    main()
