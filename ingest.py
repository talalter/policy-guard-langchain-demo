"""
ingest.py - Build the retrieval index from data/docs/*.txt files.

Supports named ingestion strategies so you can compare different chunking
and embedding configurations side-by-side. Each strategy writes to its own
directory under data/indices/{name}/.

Usage:
    python ingest.py                          # ingest with 'baseline' strategy
    python ingest.py --strategy large-chunks  # ingest with 'large-chunks' strategy
    python ingest.py --list                   # show all available strategies
    python ingest.py --docs-dir path          # use a custom docs directory
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent
DEFAULT_DOCS_DIR = REPO_ROOT / "data" / "docs"


import re


def clean_doc(text: str) -> str:
    """Strip rST heading underlines and collapse excessive whitespace."""
    # Remove lines that are purely = or - (heading underlines).
    text = re.sub(r"^[=\-]{3,}$", "", text, flags=re.MULTILINE)
    # Collapse 3+ consecutive newlines into 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_docs(docs_dir: Path) -> dict[str, str]:
    """Load all .txt files from a directory and return {name: text}."""
    if not docs_dir.exists() or not list(docs_dir.glob("*.txt")):
        logger.error("No .txt files found in %s.", docs_dir)
        sys.exit(1)

    docs: dict[str, str] = {}
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        text = clean_doc(text)
        docs[path.stem] = text
        logger.info("Loaded %s (%d chars)", path.name, len(text))
    return docs


def _list_strategies(strategies: dict, default_strategy: str) -> None:
    """Log all available ingestion strategies."""
    for name, cfg in strategies.items():
        marker = " (default)" if name == default_strategy else ""
        logger.info(
            "  %s%s: max_tokens=%d, overlap=%d, model=%s",
            name, marker, cfg.max_tokens, cfg.overlap_tokens, cfg.embedding_model,
        )


def _run_ingest(config, docs_dir: Path) -> None:
    """Load docs, chunk, and build the retrieval index for the given config."""
    from pipeline.chunker import chunk_documents
    from pipeline.retriever import Retriever

    docs = load_docs(docs_dir)
    chunks = chunk_documents(
        docs,
        max_tokens=config.max_tokens,
        overlap_tokens=config.overlap_tokens,
    )
    retriever = Retriever(
        index_dir=config.index_dir,
        embedding_model=config.embedding_model,
    )
    retriever.ingest(chunks)
    config.save()
    logger.info(
        "Done - strategy '%s': %d chunks indexed to %s",
        config.name, len(chunks), config.index_dir,
    )


def main() -> None:
    """Parse args, then list strategies or run ingestion."""
    from pipeline.ingest_config import DEFAULT_STRATEGY, STRATEGIES

    parser = argparse.ArgumentParser(description="Build retrieval index from docs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging")
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        choices=list(STRATEGIES.keys()),
        help=f"Ingestion strategy to use (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Directory containing .txt doc files (default: data/docs/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_strategies",
        help="List available strategies and exit",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_strategies:
        _list_strategies(STRATEGIES, DEFAULT_STRATEGY)
        return

    _run_ingest(STRATEGIES[args.strategy], args.docs_dir)


if __name__ == "__main__":
    main()
