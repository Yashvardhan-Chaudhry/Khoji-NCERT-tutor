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

import json
import logging
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime

import chromadb
import ollama
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from khoji import persona
from khoji.config import settings

# Dedicated logger so per-request timing lines show in the terminal regardless of
# uvicorn's --log-level. One line per request; the full record goes to EVAL.md.
log = logging.getLogger("khoji")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s khoji | %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

# --- Frozen JSON contract (UI-agnostic: React or Streamlit consume the same) ---


class Source(BaseModel):
    subject: str | None = None
    klass: str | None = None
    chapter: str | None = None
    page: int | None = None
    snippet: str
    score: float
    # internal: passed the min_score gate. exclude=True keeps it out of the public
    # contract (API/SSE); it's re-added explicitly only in the EVAL record.
    kept: bool = Field(default=True, exclude=True)


class Timings(BaseModel):
    rewrite_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    persona: bool = False  # opt-in: the re-voice is a second LLM pass (slow on CPU)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    timings: Timings
    model: str
    persona_applied: bool
    rewritten_question: str | None = None
    top_score: float | None = None  # best retrieval score seen (even when refused)
    min_score: float = settings.min_score  # the relevance floor, so a UI can show "how close"


# --- Clients + in-session memory (per-session, not long-term user memory) ---

_ollama = ollama.Client(host=settings.ollama_host)
_chroma = chromadb.PersistentClient(path=str(settings.chroma_dir))

# Ollama generation options — the latency knobs (see config.py). num_ctx shrinks
# the KV cache (RAM), num_predict caps output length (the dominant cost on CPU).
_OPTS = {"num_predict": settings.num_predict, "num_ctx": settings.num_ctx}
# Rewrite is deterministic + hard-capped so it can't ramble into a bloated question.
_REWRITE_OPTS = {"num_predict": 48, "num_ctx": settings.num_ctx, "temperature": 0.0}

# Cheap signals that a question depends on earlier turns (English + common Hinglish).
_FOLLOWUP_CUES = re.compile(
    r"\b(it|its|this|that|these|those|them|they|he|she|his|her|their|"
    r"iska|iske|isko|uska|uske|usko|inka|unka|yeh|woh|iss|uss)\b",
    re.IGNORECASE,
)
_FOLLOWUP_START = re.compile(r"^(and|so|but|also|what about|how about|why|why not|then)\b", re.IGNORECASE)

# session_id -> recent [{"q":..., "a":...}] turns; process-local, cleared on restart.
SESSIONS: dict[str, list[dict]] = defaultdict(list)
_MAX_HISTORY = 6

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Readiness log + optional warm-up so the first real query isn't cold-loaded.
    n = _corpus_count()
    if n:
        log.info("corpus ready: %d chunks in '%s'", n, settings.collection)
    else:
        log.warning("no corpus ingested — run `python -m khoji.ingest` before serving")
    if settings.warmup and n:
        try:
            _ollama.chat(model=settings.tutor_model,
                         messages=[{"role": "user", "content": "ok"}],
                         options={"num_predict": 1}, keep_alive=settings.keep_alive)
            _embed("warmup")
            log.info("warmed up %s + %s", settings.tutor_model, settings.embed_model)
        except Exception as e:  # warm-up is best-effort; serving still works cold
            log.warning("warmup skipped: %s", e)
    yield


app = FastAPI(title="Khoji — NCERT tutor", version="0.1.0", lifespan=lifespan)


def _corpus_count() -> int:
    try:
        return _chroma.get_collection(settings.collection).count()
    except Exception:
        return 0


def _meta(resp) -> dict:
    """Exact token-level metrics from an Ollama response (durations are ns).

    Works whether ollama-python hands back a dict or a ChatResponse object.
    The final chunk of a streamed response carries these same fields.
    """
    g = resp.get if isinstance(resp, dict) else (lambda k, d=None: getattr(resp, k, d))
    ec = g("eval_count", 0) or 0            # output tokens
    ed = g("eval_duration", 0) or 0         # generation time, ns
    pc = g("prompt_eval_count", 0) or 0     # prompt tokens actually processed
    ld = g("load_duration", 0) or 0         # model load time, ns (big = cold)
    return {
        "prompt_tokens": pc,
        "output_tokens": ec,
        "tok_per_s": round(ec / (ed / 1e9), 2) if ed else 0.0,
        "gen_ms": round(ed / 1e6, 1),
        "load_ms": round(ld / 1e6, 1),
    }


def _embed(text: str) -> list[float]:
    return _ollama.embeddings(model=settings.embed_model, prompt=text)["embedding"]


def _config_fingerprint() -> dict:
    """The knobs that produced a run — stamped on every eval record so results are
    attributable to a config (this is what would have caught the L2-vs-cosine mixup)."""
    space = None
    try:
        space = _chroma.get_collection(settings.collection).metadata.get("hnsw:space")
    except Exception:
        pass
    return {
        "tutor_model": settings.tutor_model,
        "embed_model": settings.embed_model,
        "space": space,
        "top_k": settings.top_k,
        "min_score": settings.min_score,
        "num_predict": settings.num_predict,
        "num_ctx": settings.num_ctx,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def _needs_rewrite(question: str, history: list[dict]) -> bool:
    """Gate: rewrite ONLY genuine context-dependent follow-ups (cheap, no LLM).

    Standalone questions ("What is an electron?") pass through untouched — that's
    the common case and was the source of the old over-rewriting: needless bloat
    and a wasted 13-25s LLM call every turn.
    """
    if not history:
        return False
    q = question.strip()
    if len(q.split()) <= 2:               # "why?", "and protons?"
        return True
    if _FOLLOWUP_START.search(q):         # "and what about neutrons"
        return True
    return bool(_FOLLOWUP_CUES.search(q))  # contains a pronoun / deictic reference


def _rewrite(question: str, history: list[dict]) -> str:
    """Resolve a follow-up into a standalone question — only when gated in.

    We retrieve on the resolved form so the vector search gets a self-contained
    query. Kept minimal: temperature 0 + a hard cap, using the recent *questions*
    (not answers) for the antecedent so the model can't latch onto an unrelated
    noun from a previous answer. Any empty/runaway rewrite falls back to original.
    """
    if not _needs_rewrite(question, history):
        return question
    convo = "\n".join(f"Q: {t['q']}" for t in history[-2:])
    prompt = (
        "Rewrite the follow-up into a standalone question by replacing references "
        "(it, this, that, iska, ...) with the specific topic from the recent "
        "questions. Output ONLY the rewritten question on one line — no explanation, "
        f"no added facts.\n\nRecent questions:\n{convo}\n\nFollow-up: {question}\n"
        "Standalone question:"
    )
    # Reuse the already-loaded tutor model with a system override — NOT base
    # phi3.5, which is a different model name and would load a 2nd ~3.9GB copy
    # into RAM (fatal paging on this box). Same weights, one resident model.
    resp = _ollama.chat(
        model=settings.tutor_model,
        messages=[
            {"role": "system", "content":
             "You rewrite a student's follow-up into ONE standalone question. "
             "Output only the rewritten question, nothing else."},
            {"role": "user", "content": prompt},
        ],
        options=_REWRITE_OPTS,
        keep_alive=settings.keep_alive,
    )
    out = resp["message"]["content"].strip().strip('"').strip()
    if not out or len(out) > 220:  # runaway or empty -> trust the original
        return question
    return out


def _retrieve(question: str) -> list[Source]:
    """Return ALL top-k candidates (Chroma-sorted), each flagged kept/rejected.

    The relevance gate lives in the `kept` flag (score >= min_score), NOT in the
    return value: callers use `_passed()` for generation/response, but the full
    list is logged so refusals aren't a black box (we can see the near-misses).
    """
    try:
        collection = _chroma.get_collection(settings.collection)
    except Exception:
        return []  # nothing ingested yet
    res = collection.query(query_embeddings=[_embed(question)], n_results=settings.top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    scored: list[Source] = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        score = round(1.0 - float(dist), 4)  # cosine distance -> similarity 0..1
        scored.append(
            Source(
                subject=meta.get("subject"),
                klass=meta.get("klass"),
                chapter=meta.get("chapter"),
                page=meta.get("page"),
                snippet=doc[:300],
                score=score,
                kept=score >= settings.min_score,
            )
        )
    if scored and not any(s.kept for s in scored):
        log.info("retrieval below threshold (top=%.3f < %.2f) — refusing",
                 max(s.score for s in scored), settings.min_score)
    return scored


def _passed(candidates: list[Source]) -> list[Source]:
    """The gate: candidates that cleared min_score (used for generation + citation)."""
    return [s for s in candidates if s.kept]


def _public_sources(candidates: list[Source]) -> list[dict]:
    """Kept sources for the UI/contract — `kept` is auto-excluded (Field exclude=True)."""
    return [s.model_dump() for s in _passed(candidates)]


def _top_score(candidates: list[Source]) -> float | None:
    """Best retrieval score seen (even if nothing cleared the gate)."""
    return max((s.score for s in candidates), default=None)


# Shown when retrieval comes back empty — we refuse to guess rather than hallucinate.
_NO_SOURCES = ("I couldn't find this in the ingested NCERT material, so I won't guess. "
               "Try rephrasing, or check that the relevant chapter has been ingested.")


def _build_prompt(question: str, sources: list[Source]) -> str:
    """Dynamic user message shared by the blocking and streaming paths.

    Only the changing content lives here (class, passages, question) — the tutor
    rules are baked into the khoji-phi Modelfile's SYSTEM, so that static prefix
    stays byte-identical across requests and Ollama can reuse its KV cache.
    """
    klass = next((s.klass for s in sources if s.klass), None)
    context = "\n\n".join(
        f"[{i}] (subject={s.subject}, class={s.klass}, chapter={s.chapter}, page={s.page})\n{s.snippet}"
        for i, s in enumerate(sources, 1)
    )
    return f"Student class: {klass or 'unknown'}\n\nPassages:\n{context}\n\nQuestion: {question}"


def _generate(question: str, sources: list[Source]) -> tuple[str, dict]:
    """Grounded answer via the tutor model. `sources` must already be gated (kept).

    Returns (answer, ollama-metrics). On an Ollama failure the answer is an explicit
    "<ERROR: ...>" marker and meta carries {"error": ...}, so a broken run is a visible
    row in EVAL rather than a silent malformed block.
    """
    if not sources:
        return _NO_SOURCES, {}
    try:
        resp = _ollama.chat(
            model=settings.tutor_model,
            messages=[{"role": "user", "content": _build_prompt(question, sources)}],
            options=_OPTS,
            keep_alive=settings.keep_alive,
        )
        return resp["message"]["content"].strip(), _meta(resp)
    except Exception as e:
        log.warning("generation failed: %s", e)
        return f"<ERROR: {e}>", {"error": str(e)}


def _oneline(text: str, n: int = 200) -> str:
    """Whitespace-collapsed, truncated snippet for the retrieval log."""
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n] + "…"


def _log_eval(rec: dict) -> None:
    """Append the run to EVAL.md (human/Claude-readable) + EVAL.jsonl (machine-parseable).

    The record is intentionally exhaustive — it's the diagnostic surface. Every top-k
    candidate is logged with its score, kept/rejected flag, and a snippet, so refusals
    and wrong-chunk retrievals (e.g. a Bohr query pulling Rutherford's bio) are visible.
    """
    if not settings.eval_log:
        return
    cfg = rec["config"]
    cfg_line = (f"tutor={cfg['tutor_model']} embed={cfg['embed_model']} space={cfg['space']} "
                f"top_k={cfg['top_k']} min_score={cfg['min_score']} "
                f"num_predict={cfg['num_predict']} num_ctx={cfg['num_ctx']} "
                f"chunk={cfg['chunk_size']}/{cfg['chunk_overlap']}")
    cand_lines = "\n".join(
        f"  - [{i}] {'✓' if c['kept'] else '✗'} {c['score']} "
        f"{c['subject']} cls{c['klass']} p{c['page']} :: \"{_oneline(c['snippet'])}\""
        for i, c in enumerate(rec["candidates"], 1)
    ) or "  - (none retrieved)"
    block = (
        f"\n## run — {rec['ts']}  ({rec['endpoint']})\n"
        f"- **config:** {cfg_line}\n"
        f"- **session:** {rec['session_id']} | rewrite_fired={'yes' if rec['rewrite_fired'] else 'no'} "
        f"| persona req={'on' if rec['persona_requested'] else 'off'} "
        f"applied={'yes' if rec['persona_applied'] else 'no'}\n"
        f"- **question:** {rec['question']}\n"
        + (f"- **rewritten:** {rec['rewritten']}\n" if rec["rewritten"] else "")
        + f"- **timings ms:** ttft={rec['ttft_ms']} rewrite={rec['rewrite_ms']} "
        f"retrieve={rec['retrieve_ms']} generate={rec['generate_ms']} total={rec['total_ms']}\n"
        f"- **tokens:** prompt={rec['prompt_tokens']} output={rec['output_tokens']} "
        f"tok/s={rec['tok_per_s']} load_ms={rec['load_ms']}\n"
        f"- **retrieval (top_k · ✓kept ✗rejected · top={rec['top_score']}):**\n{cand_lines}\n"
        + (f"- **error:** {rec['error']}\n" if rec.get("error") else "")
        + f"- **answer:**\n\n{rec['answer']}\n"
    )
    try:
        with open(settings.eval_path, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        log.warning("could not write eval report to %s", settings.eval_path)
    try:
        with open(settings.eval_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        log.warning("could not write eval jsonl to %s", settings.eval_jsonl_path)


def _emit(endpoint: str, question: str, rewritten: str, candidates: list[Source],
          answer: str, timings: "Timings", meta: dict, *, session_id: str | None = None,
          persona_requested: bool = False, persona_applied: bool = False,
          ttft_ms: float | None = None) -> None:
    """Log a one-line summary to the terminal + append the full record to EVAL.md/.jsonl."""
    passed = _passed(candidates)
    top = _top_score(candidates)
    log.info(
        "%s | ttft=%s rewrite=%.0f retrieve=%.0f generate=%.0f total=%.0f ms | "
        "out_tok=%s tok/s=%s load=%sms | kept=%d/%d top=%s refused=%s persona=%s",
        endpoint, f"{ttft_ms:.0f}" if ttft_ms is not None else "-",
        timings.rewrite_ms, timings.retrieve_ms, timings.generate_ms, timings.total_ms,
        meta.get("output_tokens"), meta.get("tok_per_s"), meta.get("load_ms"),
        len(passed), len(candidates), top, not passed, persona_applied,
    )
    _log_eval({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "endpoint": endpoint, "model": settings.tutor_model,
        "config": _config_fingerprint(),
        "session_id": session_id,
        "rewrite_fired": rewritten != question,
        "persona_requested": persona_requested, "persona_applied": persona_applied,
        "question": question,
        "rewritten": rewritten if rewritten != question else None,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "rewrite_ms": timings.rewrite_ms, "retrieve_ms": timings.retrieve_ms,
        "generate_ms": timings.generate_ms, "total_ms": timings.total_ms,
        "prompt_tokens": meta.get("prompt_tokens"), "output_tokens": meta.get("output_tokens"),
        "tok_per_s": meta.get("tok_per_s"), "load_ms": meta.get("load_ms"),
        "top_score": top, "kept_count": len(passed),
        "candidates": [{**s.model_dump(), "kept": s.kept} for s in candidates],
        "error": meta.get("error"),
        "answer": answer,
    })


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "collection": settings.collection, "chunks": _corpus_count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    history = SESSIONS[req.session_id] if req.session_id else []

    t1 = time.perf_counter()
    rewritten = _rewrite(req.question, history)
    t2 = time.perf_counter()

    candidates = _retrieve(rewritten)
    passed = _passed(candidates)
    t3 = time.perf_counter()

    answer, meta = _generate(rewritten, passed)
    t4 = time.perf_counter()

    persona_applied = False
    if req.persona and passed:
        try:
            answer = persona.revoice(answer)
            persona_applied = True
        except Exception:
            pass  # persona is delivery-only; a plain correct answer still ships

    if req.session_id:
        history.append({"q": req.question, "a": answer})
        del history[:-_MAX_HISTORY]  # keep the tail bounded

    timings = Timings(
        rewrite_ms=round((t2 - t1) * 1000, 1),
        retrieve_ms=round((t3 - t2) * 1000, 1),
        generate_ms=round((t4 - t3) * 1000, 1),
        total_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    _emit("/ask", req.question, rewritten, candidates, answer, timings, meta,
          session_id=req.session_id, persona_requested=req.persona,
          persona_applied=persona_applied)
    return AskResponse(
        answer=answer,
        sources=passed,
        timings=timings,
        model=settings.tutor_model,
        persona_applied=persona_applied,
        rewritten_question=rewritten if rewritten != req.question else None,
        top_score=_top_score(candidates),
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-Sent Events variant: citations arrive first, then tokens stream.

    The point is time-to-first-token — the student sees sources in ~seconds and
    the answer typing out, instead of staring at a frozen page for minutes.
    Persona is not applied here (a second pass can't stream cleanly).
    """
    def gen():
        t0 = time.perf_counter()
        history = SESSIONS[req.session_id] if req.session_id else []

        t1 = time.perf_counter()
        rewritten = _rewrite(req.question, history)
        t2 = time.perf_counter()
        candidates = _retrieve(rewritten)
        passed = _passed(candidates)
        t3 = time.perf_counter()

        # 1) meta first — UI renders citations before a single token is generated.
        #    top_score is included so the client can explain a refusal (how close it was).
        yield _sse("meta", {
            "sources": _public_sources(candidates),
            "top_score": _top_score(candidates),
            "min_score": settings.min_score,
            "rewritten_question": rewritten if rewritten != req.question else None,
            "model": settings.tutor_model,
        })

        # 2) stream the grounded answer token by token
        parts: list[str] = []
        meta: dict = {}
        ttft_ms: float | None = None
        if not passed:
            parts.append(_NO_SOURCES)
            yield _sse("token", {"text": _NO_SOURCES})
        else:
            try:
                stream = _ollama.chat(
                    model=settings.tutor_model,
                    messages=[{"role": "user", "content": _build_prompt(rewritten, passed)}],
                    options=_OPTS,
                    keep_alive=settings.keep_alive,
                    stream=True,
                )
                for chunk in stream:
                    tok = chunk["message"]["content"]
                    if tok:
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - t0) * 1000, 1)
                        parts.append(tok)
                        yield _sse("token", {"text": tok})
                    meta = _meta(chunk)  # final chunk (done=True) carries the metrics
            except Exception as e:
                log.warning("stream generation failed: %s", e)
                meta = {"error": str(e)}
                parts = [f"<ERROR: {e}>"]
                yield _sse("token", {"text": parts[0]})
        answer = "".join(parts).strip()
        t4 = time.perf_counter()

        if req.session_id:
            history.append({"q": req.question, "a": answer})
            del history[:-_MAX_HISTORY]

        timings = Timings(
            rewrite_ms=round((t2 - t1) * 1000, 1),
            retrieve_ms=round((t3 - t2) * 1000, 1),
            generate_ms=round((t4 - t3) * 1000, 1),
            total_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        _emit("/ask/stream", req.question, rewritten, candidates, answer, timings, meta,
              session_id=req.session_id, persona_requested=req.persona,
              persona_applied=False, ttft_ms=ttft_ms)

        # 3) done — timings + persona_applied for the same contract as /ask
        yield _sse("done", {"timings": timings.model_dump(), "meta": meta,
                            "persona_applied": False})

    return StreamingResponse(gen(), media_type="text/event-stream")
