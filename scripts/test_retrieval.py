"""
test_retrieval.py - Manually inspect what the retriever returns for any query.

Useful for verifying that the FAISS + BM25 index contains relevant chunks
and that the re-ranker is surfacing the right results.

Usage:
    python eval/test_retrieval.py "What is the rate limit for free plan users?"
    python eval/test_retrieval.py "How does Fynlo authenticate requests?" --top-k 3
    python eval/test_retrieval.py "What headers are required on every request?" --full
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pipeline.ingest_config import DEFAULT_STRATEGY, STRATEGIES
from pipeline.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _display_results(chunks: list[str], full: bool) -> None:
    """Log retrieved chunks with optional truncation."""
    logger.info("=" * 70)
    for i, chunk in enumerate(chunks, start=1):
        text = chunk if full else chunk[:400]
        ellipsis = "..." if not full and len(chunk) > 400 else ""
        logger.info("\n[%d] %s%s", i, text, ellipsis)
        logger.info("    (%d chars total)", len(chunk))
        logger.info("-" * 70)


def main() -> None:
    """Load the retriever and log top-k chunks for the given query."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Query to test")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--full", action="store_true", help="Print full chunk text (default: first 400 chars)")
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        choices=list(STRATEGIES.keys()),
        help=f"Which strategy's index to query (default: {DEFAULT_STRATEGY})",
    )
    args = parser.parse_args()

    config = STRATEGIES[args.strategy]
    retriever = Retriever(
        index_dir=config.index_dir,
        embedding_model=config.embedding_model,
    )
    chunks = retriever.retrieve(args.query, top_k=args.top_k)

    logger.info("\nQuery: %s", args.query)
    logger.info("Results: %d chunks", len(chunks))
    _display_results(chunks, args.full)


if __name__ == "__main__":
    sys.exit(main())
