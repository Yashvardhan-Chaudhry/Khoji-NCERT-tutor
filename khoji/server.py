"""Serving — the online (real-time) phase, hand-written on purpose.

This is Khoji's signature: the retrieve -> prompt -> generate -> cite loop is
explicit here, not hidden inside a LangChain chain, because provenance is the
whole point. It reads the same Chroma store that ingest.py wrote.

The loop per question:
    1. session memory  -> if this is a follow-up, rewrite it to a standalone
                          question using the last few turns ("iska example do"
                          -> "give an example of photosynthesis")
    2. retrieve        -> embed the (rewritten) question, top-k from Chroma
    3. generate        -> Phi-3.5 answers grounded ONLY in the retrieved passages
    4. cite            -> return the sources (chapter/page) alongside the answer
    5. persona         -> re-voice as Khoji Bhai (facts untouched; toggleable)

Boot:  uvicorn khoji.server:app --reload
"""

from __future__ import annotations

import time
from collections import defaultdict

import chromadb
import ollama
from fastapi import FastAPI
from pydantic import BaseModel, Field

from khoji import persona
from khoji.config import settings

# --- Frozen JSON contract (UI-agnostic: React or Streamlit consume the same) ---


class Source(BaseModel):
    subject: str | None = None
    klass: str | None = None
    chapter: str | None = None
    page: int | None = None
    snippet: str
    score: float


class Timings(BaseModel):
    rewrite_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    persona: bool = True


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    timings: Timings
    model: str
    persona_applied: bool
    rewritten_question: str | None = None


# --- Clients + in-session memory (per-session, not long-term user memory) ---

_ollama = ollama.Client(host=settings.ollama_host)
_chroma = chromadb.PersistentClient(path=str(settings.chroma_dir))

# session_id -> recent [{"q":..., "a":...}] turns; process-local, cleared on restart.
SESSIONS: dict[str, list[dict]] = defaultdict(list)
_MAX_HISTORY = 6

app = FastAPI(title="Khoji — NCERT tutor", version="0.1.0")


def _embed(text: str) -> list[float]:
    return _ollama.embeddings(model=settings.embed_model, prompt=text)["embedding"]


def _rewrite(question: str, history: list[dict]) -> str:
    """Turn a context-dependent follow-up into a standalone question.

    No history -> return as-is. This is what makes conversational context work
    without polluting retrieval: we retrieve on the resolved standalone form.
    """
    if not history:
        return question
    convo = "\n".join(f"Student: {t['q']}\nKhoji: {t['a']}" for t in history[-3:])
    prompt = (
        "Rewrite the student's latest message into a single standalone question "
        "that makes sense without the earlier conversation. Resolve pronouns and "
        "references like 'iska', 'that', 'it'. Reply with ONLY the rewritten "
        f"question, nothing else.\n\nConversation:\n{convo}\n\nLatest: {question}"
    )
    resp = _ollama.chat(model=settings.gen_model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip() or question


def _retrieve(question: str) -> list[Source]:
    try:
        collection = _chroma.get_collection(settings.collection)
    except Exception:
        return []  # nothing ingested yet
    res = collection.query(query_embeddings=[_embed(question)], n_results=settings.top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    sources: list[Source] = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        sources.append(
            Source(
                subject=meta.get("subject"),
                klass=meta.get("klass"),
                chapter=meta.get("chapter"),
                page=meta.get("page"),
                snippet=doc[:300],
                score=round(1.0 / (1.0 + float(dist)), 4),  # distance -> 0..1 similarity
            )
        )
    return sources


def _generate(question: str, sources: list[Source]) -> str:
    if not sources:
        return ("I couldn't find this in the ingested NCERT material, so I won't guess. "
                "Try rephrasing, or check that the relevant chapter has been ingested.")
    klass = next((s.klass for s in sources if s.klass), None)
    context = "\n\n".join(
        f"[{i}] (subject={s.subject}, class={s.klass}, chapter={s.chapter}, page={s.page})\n{s.snippet}"
        for i, s in enumerate(sources, 1)
    )
    level = f"a Class {klass} student" if klass else "a school student"
    prompt = (
        f"You are an NCERT tutor. Answer the question for {level}, using ONLY the "
        "passages below. If the passages don't contain the answer, say so plainly "
        "instead of guessing. Cite the passages you use like [1], [2].\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}"
    )
    resp = _ollama.chat(model=settings.gen_model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    history = SESSIONS[req.session_id] if req.session_id else []

    t1 = time.perf_counter()
    rewritten = _rewrite(req.question, history)
    t2 = time.perf_counter()

    sources = _retrieve(rewritten)
    t3 = time.perf_counter()

    answer = _generate(rewritten, sources)
    t4 = time.perf_counter()

    persona_applied = False
    if req.persona and sources:
        try:
            answer = persona.revoice(answer)
            persona_applied = True
        except Exception:
            pass  # persona is delivery-only; a plain correct answer still ships

    if req.session_id:
        history.append({"q": req.question, "a": answer})
        del history[:-_MAX_HISTORY]  # keep the tail bounded

    return AskResponse(
        answer=answer,
        sources=sources,
        timings=Timings(
            rewrite_ms=round((t2 - t1) * 1000, 1),
            retrieve_ms=round((t3 - t2) * 1000, 1),
            generate_ms=round((t4 - t3) * 1000, 1),
            total_ms=round((time.perf_counter() - t0) * 1000, 1),
        ),
        model=settings.gen_model,
        persona_applied=persona_applied,
        rewritten_question=rewritten if rewritten != req.question else None,
    )
