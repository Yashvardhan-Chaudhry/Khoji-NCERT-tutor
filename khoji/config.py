"""Khoji settings — the single source for model ids, paths, and retrieval knobs.

Both phases import from here: `ingest.py` writes the vector store, `server.py`
reads it. Keeping this in one place is what lets ingestion and serving stay
decoupled around the Chroma seam.

Everything has an offline-sane default; override via environment variables
(e.g. OLLAMA_HOST) without touching code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of this package dir. Paths below hang off it so the app
# runs the same regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KHOJI_", env_file=".env", extra="ignore")

    # --- Ollama / models (all served locally; both already pulled) ---
    ollama_host: str = "http://localhost:11434"
    gen_model: str = "phi3.5:3.8b"        # base Phi (benchmark ref); serving reuses tutor_model
    tutor_model: str = "khoji-phi"        # answers AND rewrite: one resident model, SYSTEM overridden per task
    bench_model: str = "qwen2.5:3b"       # honest benchmark foil (see IMPROVEMENTS.md)
    embed_model: str = "nomic-embed-text"  # ingestion + query embeddings

    # --- Storage (created on demand; both gitignored) ---
    raw_dir: Path = ROOT / "data" / "raw"        # source NCERT PDFs live here
    chroma_dir: Path = ROOT / "data" / "chroma"  # persisted vector store
    collection: str = "ncert"

    # --- Retrieval / chunking (tuned during the Session-2 ingestion spike) ---
    top_k: int = 3                # 3 keeps prompts short + answers focused (5 diluted quality)
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_score: float = 0.55       # cosine floor; tunable. RE-TUNE after prefixes shift the scale

    # --- Hybrid retrieval: dense (cosine) ∪ BM25 (lexical) → RRF pool → cross-encoder rerank ---
    hybrid: bool = True
    pool_size: int = 10           # dense + BM25 candidates fetched before fusion
    rrf_k: int = 60               # reciprocal-rank-fusion constant

    # Cross-encoder reranker over the fused pool — the precision stage (Bohr↔Rutherford fix).
    rerank: bool = True
    rerank_backend: str = "flashrank"                 # swappable; flashrank = ONNX, torch-free
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"     # ~35MB; bge-reranker-v2-m3 = heavier upgrade
    rerank_cache_dir: Path = ROOT / "data" / "rerank"  # model cache (gitignored under data/)
    min_rerank_score: float = 0.30    # refuse floor on rerank score — PLACEHOLDER, tune via EVAL
    # nomic-embed-text task prefixes — REQUIRED for good retrieval (asymmetric
    # query/document encoding). Applied to queries at serve time, documents at ingest.
    query_prefix: str = "search_query: "
    doc_prefix: str = "search_document: "

    # --- Inference latency knobs (this box is CPU-only + RAM-starved) ---
    num_predict: int = 220        # hard ceiling on answer tokens (~120 words + citations)
    num_ctx: int = 2048           # smaller context = less KV-cache RAM + prompt work
    keep_alive: str = "10m"       # keep model warm within a session; never "-1" here
    warmup: bool = True           # preload models on server boot so 1st query isn't cold

    # --- Eval report (server appends one block per request; kept local) ---
    # EVAL.md is the human-readable (Claude-readable) log; EVAL.jsonl is the flat
    # machine-parseable sidecar for scripted aggregation. Both gitignored.
    eval_log: bool = True
    eval_path: Path = ROOT / "EVAL.md"
    eval_jsonl_path: Path = ROOT / "EVAL.jsonl"

    def ensure_dirs(self) -> None:
        """Make the data dirs exist so a fresh clone works with no setup."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
