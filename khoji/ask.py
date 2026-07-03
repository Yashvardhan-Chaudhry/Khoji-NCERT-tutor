"""Manual test client — drive the server from your terminal during eval runs.

Streams /ask/stream by default so you see time-to-first-token live; prints a
one-line timing summary at the end. The server writes the full record to EVAL.md
either way, so plain curl.exe works too — this is just a nicer loop.

    python -m khoji.ask "What is photosynthesis?"
    python -m khoji.ask "iska example do" --session s1
    python -m khoji.ask "What is a cell?" --no-stream      # blocking /ask
    python -m khoji.ask "What is a cell?" --persona         # opt into Khoji Bhai voice
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _post(url: str, payload: dict):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    return urllib.request.urlopen(req)


def _run_stream(base: str, payload: dict) -> None:
    t0 = time.perf_counter()
    meta_t = first_tok_t = None
    ev = None
    with _post(base + "/ask/stream", payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
                if ev == "meta":
                    meta_t = time.perf_counter() - t0
                    _print_sources(data.get("sources", []))
                    if data.get("rewritten_question"):
                        print(f"(rewritten -> {data['rewritten_question']})\n")
                    print("answer: ", end="", flush=True)
                elif ev == "token":
                    if first_tok_t is None:
                        first_tok_t = time.perf_counter() - t0
                    print(data["text"], end="", flush=True)
                elif ev == "done":
                    tim = data["timings"]
                    m = data.get("meta", {})
                    total = time.perf_counter() - t0
                    print("\n")
                    print(f"[TTFT {first_tok_t or 0:.1f}s | meta {meta_t or 0:.1f}s | "
                          f"total {total:.1f}s | server generate {tim['generate_ms']/1000:.1f}s | "
                          f"out_tok {m.get('output_tokens')} @ {m.get('tok_per_s')} tok/s | "
                          f"load {m.get('load_ms')}ms]")


def _run_blocking(base: str, payload: dict) -> None:
    t0 = time.perf_counter()
    with _post(base + "/ask", payload) as r:
        resp = json.loads(r.read())
    total = time.perf_counter() - t0
    _print_sources([s for s in resp["sources"]])
    if resp.get("rewritten_question"):
        print(f"(rewritten -> {resp['rewritten_question']})\n")
    print("answer:\n" + resp["answer"] + "\n")
    tim = resp["timings"]
    print(f"[wall {total:.1f}s | rewrite {tim['rewrite_ms']/1000:.1f}s | "
          f"retrieve {tim['retrieve_ms']/1000:.1f}s | generate {tim['generate_ms']/1000:.1f}s | "
          f"persona_applied {resp['persona_applied']}]")


def _print_sources(sources: list[dict]) -> None:
    if not sources:
        print("sources: (none)\n")
        return
    print("sources:")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s.get('subject')} class {s.get('klass')} "
              f"ch={s.get('chapter')} p{s.get('page')} score={s.get('score')}")
    print()


def main() -> None:
    # Windows consoles default to cp1252 and crash on Greek/math chars (α, °, ×)
    # that science answers are full of. Force UTF-8, never crash on output.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Ask Khoji from the terminal.")
    ap.add_argument("question")
    ap.add_argument("--session", default=None, help="session id (enables follow-up rewrite)")
    ap.add_argument("--no-stream", action="store_true", help="use blocking /ask")
    ap.add_argument("--persona", action="store_true", help="opt into the Khoji Bhai re-voice")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"
    payload = {"question": args.question, "session_id": args.session, "persona": args.persona}
    try:
        if args.no_stream:
            _run_blocking(base, payload)
        else:
            _run_stream(base, payload)
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach the server at {base} — is uvicorn running? ({e})")


if __name__ == "__main__":
    main()
