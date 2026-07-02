# Khoji — an offline-first NCERT tutor

**Khoji** (*"khoj" = discovery*) is a local, grounded, cited tutor for the NCERT
curriculum. A student asks a question; Khoji retrieves the relevant NCERT passage,
answers in age-appropriate language, and **cites the chapter and page**. A persona
layer (*Khoji Bhai*) re-voices the delivery in warm Hinglish — **without ever
touching the retrieved facts**.

Built to run **fully offline** (Ollama + a local vector DB + FastAPI on localhost),
so it works in schools with limited internet. "Online" here means real-time
serving, not internet-connected.

## How it works

Two phases, joined by the vector DB as a frozen contract:

```
INGESTION (offline, once per corpus)      SERVING (real-time)
  NCERT PDF                                 POST /ask
   -> load + split (LangChain, PyMuPDF)      -> rewrite follow-up to standalone Q
   -> embed (nomic-embed-text / Ollama)      -> embed + retrieve top-k (Chroma)
   -> store in Chroma                        -> generate grounded answer (Phi-3.5)
                                             -> cite chapter/page
                                             -> re-voice as Khoji Bhai (facts intact)
```

The retrieval + citation loop (`server.py`) is written by hand — provenance is the
point, so it isn't buried in a framework. Only the ingestion plumbing uses LangChain.

## Stack

- **LLM:** Phi-3.5 (`phi3.5:3.8b`), served locally by **Ollama**
- **Embeddings:** `nomic-embed-text`
- **Vector DB:** **Chroma** (embedded, persists to `data/chroma`)
- **Backend:** **FastAPI**
- **Ingestion:** LangChain + PyMuPDF

All open-source.

## Quick start

Prerequisites: [Ollama](https://ollama.com) running, with the models pulled:

```bash
ollama pull phi3.5:3.8b
ollama pull nomic-embed-text
```

Install and run:

```bash
pip install -r requirements.txt

# 1. Ingest a NCERT PDF (drop your PDFs in data/raw/)
python -m khoji.ingest --pdf data/raw/science_viii.pdf --subject science --klass 8

# 2. Serve
uvicorn khoji.server:app --reload
```

Ask a question:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What is photosynthesis?","session_id":"s1"}'
```

The response is a UI-agnostic JSON contract — `answer`, `sources`
(subject/class/chapter/page), `timings`, `model`, `persona_applied`, and
`rewritten_question` when a follow-up was resolved from session context.

## Status

Early scaffold — the pipeline runs end-to-end; retrieval quality is being tuned
against a real NCERT corpus. A benchmark (Phi-3.5 vs Qwen2.5) and a chat frontend
are on the roadmap.
