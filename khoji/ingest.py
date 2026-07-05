"""Ingestion — the offline, run-once-per-corpus phase.

This is the *boring plumbing* half of Khoji, so it leans on LangChain: load a
NCERT PDF, split it into citation-sized chunks carrying chapter/page metadata,
embed each chunk with nomic-embed-text (via Ollama), and persist to Chroma.

The serving side (server.py) deliberately does NOT use LangChain — that's where
the retrieve->cite discipline is hand-written and on display. Here, nobody
learns anything hand-rolling a PDF loader, so we don't.

Usage:
    python -m khoji.ingest --pdf data/raw/science_viii.pdf --subject science --klass 8
    python -m khoji.ingest --pdf data/raw/science_viii.pdf --subject science --klass 8 --chapter "Crop Production"

The vector store is the frozen contract between this and serving. Re-running
ingest for the same PDF appends; wipe data/chroma to rebuild clean.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from khoji.config import settings


def _degarble(text: str) -> str:
    """Collapse consecutive repeated tokens / short phrases from PDF extraction.

    NCERT's decorative drop-caps and boxed headings come out as runs like
    "BOHR OHR OHR OHR'S MODEL MODEL MODEL" or "in Different Orbits (Shells)?" ×5.
    We collapse any phrase of 1..6 words that repeats immediately, keeping one copy.
    Token-based (not regex) so there's no catastrophic-backtracking risk.
    """
    words = text.split()
    out: list[str] = []
    i, n = 0, len(words)
    while i < n:
        collapsed = False
        for length in range(1, min(6, (n - i) // 2) + 1):
            phrase = words[i:i + length]
            reps = 1
            while words[i + reps * length: i + (reps + 1) * length] == phrase:
                reps += 1
            if reps > 1:
                out.extend(phrase)          # keep a single copy
                i += reps * length
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    cleaned = " ".join(out)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


class _PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """nomic-embed-text needs task prefixes. Documents are embedded with the
    'search_document: ' prefix, but the STORED chunk text stays original — Chroma
    stores the page_content it passes in and embeds the (prefixed) copy we return,
    so the prefix never pollutes the retrieved snippet/context.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([settings.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(settings.query_prefix + text)


def _embeddings() -> OllamaEmbeddings:
    return _PrefixedOllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_host)


def ingest_pdf(pdf: Path, subject: str, klass: str, chapter: str | None = None,
               reset: bool = False) -> int:
    """Load one PDF into the Chroma store. Returns the number of chunks written.

    Every chunk keeps enough metadata to build an honest citation later:
    subject, class, chapter (optional), 1-indexed page, and source filename.

    `reset=True` clears the collection first — ingest otherwise *appends*, so
    re-ingesting the same chapter would silently duplicate chunks.
    """
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    settings.ensure_dirs()

    # One shared client so the optional reset and the write don't open two
    # PersistentClients on the same path.
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:
            pass  # nothing to reset yet

    # One Document per page; PyMuPDF fills metadata['page'] (0-indexed) + 'source'.
    pages = PyMuPDFLoader(str(pdf)).load()

    # Light de-garble: NCERT PDFs extract decorative headers as repeated junk
    # ("OHR OHR OHR", "(Shells)? (Shells)?", "4.2.3 4.2.3 4.2.3"), which buries the
    # real facts and hurts both embedding and the model. Collapse the repeats.
    for p in pages:
        p.page_content = _degarble(p.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    # Normalize + enrich metadata so serving can cite chapter/page directly.
    for c in chunks:
        c.metadata["subject"] = subject
        c.metadata["klass"] = str(klass)
        if chapter:
            c.metadata["chapter"] = chapter
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1  # 1-indexed for humans
        c.metadata["source"] = pdf.name

    store = Chroma(
        collection_name=settings.collection,
        embedding_function=_embeddings(),
        client=client,
        collection_metadata={"hnsw:space": "cosine"},  # meaningful 0-1 similarity scores
    )
    store.add_documents(chunks)

    return len(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a NCERT PDF into Khoji's vector store.")
    ap.add_argument("--pdf", required=True, type=Path, help="Path to the source PDF")
    ap.add_argument("--subject", required=True, help="e.g. science, math, history")
    ap.add_argument("--klass", required=True, help="Class number, e.g. 8 (V-XII)")
    ap.add_argument("--chapter", default=None, help="Optional chapter name for citations")
    ap.add_argument("--reset", action="store_true",
                    help="Clear the collection first (avoids duplicate chunks on re-ingest)")
    args = ap.parse_args()

    n = ingest_pdf(args.pdf, args.subject, args.klass, args.chapter, reset=args.reset)
    print(f"Ingested {n} chunks from {args.pdf.name} "
          f"(subject={args.subject}, class={args.klass}) -> {settings.chroma_dir}")


if __name__ == "__main__":
    main()
