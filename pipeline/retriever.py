"""
retriever.py - Hybrid BM25 + dense retrieval with cross-encoder re-ranking.

Two-stage retrieval pipeline:
  Stage 1 - Recall:   BM25 top-20 + dense (FAISS) top-20 → fused via RRF(k=60)
  Stage 2 - Precision: Cross-encoder re-ranks the merged candidates → top-k

Why hybrid + RRF?
  Dense-only retrieval misses exact keyword matches (version numbers, HTTP
  verbs, error codes). BM25 catches these. RRF combines both ranked lists
  without requiring their scores to be on the same scale - used in production
  by Cohere, Elasticsearch, and Weaviate.

Why cross-encoder re-ranking?
  The bi-encoder computes (query, chunk) similarity as a dot product - fast
  but imprecise. The cross-encoder sees the full pair with cross-attention,
  making far more accurate relevance judgments. Running it only on the top-40
  candidates keeps latency acceptable.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import faiss # type: ignore
import numpy as np # type: ignore 
from rank_bm25 import BM25Okapi # type: ignore
from sentence_transformers import CrossEncoder, SentenceTransformer # type: ignore

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "indices" / "baseline"

BI_ENCODER_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Stage-1 candidate count per retriever before fusion.
STAGE1_TOP_K = 20
# RRF smoothing constant - standard value from the original RRF paper.
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into tokens for BM25 indexing."""
    return text.lower().split()


def _build_faiss_index(
    bi_encoder: SentenceTransformer,
    chunks: list[str],
) -> tuple:
    """Encode chunks with the bi-encoder and build a FAISS flat-IP index.

    Embeddings are L2-normalized so inner-product search equals cosine similarity.
    Returns (index, embedding_dim).
    """
    embeddings = bi_encoder.encode(
        chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    dim = embeddings.shape[1]  # type: ignore
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype=np.float32))  # type: ignore
    return index, dim


def _rerank(
    cross_encoder: CrossEncoder,
    query: str,
    chunks: list[str],
    candidates: list[int],
    top_k: int,
) -> list[int]:
    """Re-rank candidate indices with the cross-encoder and return the top-k."""
    pairs = [(query, chunks[i]) for i in candidates]
    scores = cross_encoder.predict(pairs)  # type: ignore
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:top_k]]


def _rrf_scores(
    bm25_indices: list[int],
    dense_indices: list[int],
    k: int = RRF_K,
) -> dict[int, float]:
    """Compute Reciprocal Rank Fusion scores for two ranked lists.

    RRF score for document d = Σ 1 / (k + rank(d, list))
    Documents missing from a list are not penalized - they simply don't
    contribute to that term. This makes RRF robust to partial overlap.
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(dense_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    """Hybrid BM25 + dense retriever with cross-encoder re-ranking.

    Call ingest() once to build and persist the indices.
    Call retrieve() at query time.
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialize model objects. Indices are loaded lazily on first retrieve().

        Args:
            index_dir: Directory to read/write index files. Defaults to
                       data/indices/baseline/.
            embedding_model: Bi-encoder model name. Defaults to env
                             EMBEDDING_MODEL or all-MiniLM-L6-v2.
        """
        self._index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        bi_model = embedding_model or BI_ENCODER_MODEL
        logger.info("Loading bi-encoder: %s", bi_model)
        self._bi_encoder = SentenceTransformer(bi_model)
        logger.info("Loading cross-encoder: %s", RERANKER_MODEL)
        self._cross_encoder = CrossEncoder(RERANKER_MODEL)
        self._chunks: Optional[list[str]] = None
        self._bm25: Optional[BM25Okapi] = None
        self._faiss_index: Optional[faiss.IndexFlatIP] = None

    def ingest(self, chunks: list[str]) -> None:
        """Build BM25 and FAISS indices from chunks and persist to disk."""
        logger.info("Ingesting %d chunks...", len(chunks))
        self._chunks = chunks

        tokenized = [_tokenize(c) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

        self._faiss_index, dim = _build_faiss_index(self._bi_encoder, chunks)

        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = self._index_dir / "index.faiss"
        bm25_path = self._index_dir / "bm25.pkl"
        faiss.write_index(self._faiss_index, str(faiss_path))
        with bm25_path.open("wb") as f:
            pickle.dump((self._bm25, self._chunks), f)

        logger.info(
            "Ingestion complete. FAISS index: %d vectors, dim=%d. Saved to %s",
            self._faiss_index.ntotal,  # type: ignore
            dim,
            self._index_dir,
        )

    def _load_indices(self) -> None:
        """Load persisted indices from disk."""
        faiss_path = self._index_dir / "index.faiss"
        bm25_path = self._index_dir / "bm25.pkl"
        if not faiss_path.exists() or not bm25_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {self._index_dir}. "
                "Run ingest.py first."
            )
        self._faiss_index = faiss.read_index(str(faiss_path))
        with bm25_path.open("rb") as f:
            self._bm25, self._chunks = pickle.load(f)
        logger.info("Loaded indices from %s: %d chunks", self._index_dir, len(self._chunks)) # type: ignore

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve the top-k most relevant chunks for query.

        Pipeline: BM25 top-20 + dense top-20 → RRF → cross-encoder → top-k.
        """
        if self._chunks is None:
            self._load_indices()

        # Stage 1a: BM25 retrieval.
        bm25_scores = self._bm25.get_scores(_tokenize(query))  # type: ignore
        bm25_top = np.argsort(bm25_scores)[::-1][:STAGE1_TOP_K].tolist()

        # Stage 1b: Dense retrieval.
        query_emb = self._bi_encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)  # type: ignore
        _, dense_top_arr = self._faiss_index.search(query_emb, STAGE1_TOP_K)  # type: ignore
        dense_top = dense_top_arr[0].tolist()

        # Stage 1c: RRF fusion - merge both ranked lists.
        fused = _rrf_scores(bm25_top, dense_top)
        candidates = sorted(fused, key=lambda i: fused[i], reverse=True)

        # Stage 2: Cross-encoder re-ranking.
        top_indices = _rerank(self._cross_encoder, query, self._chunks, candidates, top_k)  # type: ignore
        logger.debug(
            "retrieve('%s'): BM25=%d, dense=%d, fused=%d, returning top %d",
            query[:50], len(bm25_top), len(dense_top), len(candidates), top_k,
        )
        return [self._chunks[i] for i in top_indices]  # type: ignore
