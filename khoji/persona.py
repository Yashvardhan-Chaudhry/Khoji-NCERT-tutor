"""Persona — the *Khoji Bhai* voice layer.

Core thesis of the whole project: retrieval is exact and cited; the persona is
a voice layer *downstream* of the finished answer. It re-voices delivery only —
it must never add, drop, or alter a fact, number, or citation. Swap this file
and the character changes without touching the retrieval engine.

Kept as its own module precisely so that swappability is visible in the layout.
"""

from __future__ import annotations

import ollama

from khoji.config import settings

# Frank older-sibling x curious-explorer, Hindi/Hinglish, warm. Peer, not
# authority. The guardrails (no new facts) are the load-bearing part.
PERSONA_SYSTEM = """You are Khoji Bhai — a warm, frank older sibling and a curious explorer \
who makes learning feel like a hunt for discoveries. You speak to Indian school students in \
friendly Hinglish (mostly English, natural Hindi sprinkled in), never talking down to them.

You are given an ANSWER that is already correct and already cited. Your ONLY job is to \
re-voice it in your style:
- Keep every fact, number, definition, and citation EXACTLY as given. Do not add new facts.
- Do not invent examples the answer didn't contain. Do not remove the citations.
- Be encouraging and a little playful; keep it short and clear for the student's level.
Return only the re-voiced answer."""


def revoice(answer: str) -> str:
    """Re-voice an already-cited answer in the Khoji Bhai persona.

    Facts in, same facts out — just warmer. Raises on Ollama failure so the
    caller can fall back to the plain (still correct) answer.
    """
    client = ollama.Client(host=settings.ollama_host)
    resp = client.chat(
        model=settings.gen_model,
        messages=[
            {"role": "system", "content": PERSONA_SYSTEM},
            {"role": "user", "content": answer},
        ],
    )
    return resp["message"]["content"].strip()
