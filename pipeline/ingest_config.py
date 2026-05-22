"""
ingest_config.py - Configuration for ingestion strategies.

Each IngestConfig fully describes a reproducible ingestion experiment:
chunking approach, token limits, embedding model. Configs are serialized
to JSON alongside the indices so any result can be traced back to its
exact settings.

Pre-defined strategies live in STRATEGIES - add new ones there.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
INDICES_DIR = REPO_ROOT / "data" / "indices"
DEFAULT_STRATEGY = "baseline"


@dataclass(frozen=True)
class IngestConfig:
    """Immutable description of an ingestion strategy."""

    name: str
    max_tokens: int
    overlap_tokens: int
    embedding_model: str

    @property
    def index_dir(self) -> Path:
        """Return the directory where this strategy's indices are stored."""
        return INDICES_DIR / self.name

    def save(self) -> None:
        """Persist this config as config.json inside its index directory."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        path = self.index_dir / "config.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info("Saved config to %s", path)

    @classmethod
    def load(cls, index_dir: Path) -> "IngestConfig":
        """Load a config from a strategy's index directory."""
        path = index_dir / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"No config.json in {index_dir}")
        data = json.loads(path.read_text())
        return cls(**data)


# ── Pre-defined strategies ──────────────────────────────────────────────

STRATEGIES: dict[str, IngestConfig] = {
    "baseline": IngestConfig(
        name="baseline",
        max_tokens=300,
        overlap_tokens=50,
        embedding_model="all-MiniLM-L6-v2",
    ),
    "large-chunks": IngestConfig(
        name="large-chunks",
        max_tokens=500,
        overlap_tokens=100,
        embedding_model="all-MiniLM-L6-v2",
    ),
    "small-chunks": IngestConfig(
        name="small-chunks",
        max_tokens=150,
        overlap_tokens=30,
        embedding_model="all-MiniLM-L6-v2",
    ),
}
