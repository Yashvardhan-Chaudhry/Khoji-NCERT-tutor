"""Manual test client — drive the server from your terminal during eval runs.

Run with no question for an interactive chat loop; pass a question for a one-shot.
Streams /ask/stream by default so you see time-to-first-token live; prints a
one-line timing summary at the end. The server writes the full record to EVAL.md
either way, so plain curl.exe works too — this is just a nicer loop.

    python -m khoji.ask                                    # interactive chat loop
    python -m khoji.ask "What is photosynthesis?"          # one-shot
    python -m khoji.ask "iska example do" --session s1      # resume a session
    python -m khoji.ask "What is a cell?" --no-stream       # blocking /ask
    python -m khoji.ask "What is a cell?" --persona          # opt into Khoji Bhai voice
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

_COLOR = False  # set once in main(); gates the dim styling


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if _COLOR else s


def _post(url: str, payload: dict):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    return urllib.request.urlopen(req)


def _run_stream(base: str, payload: dict) -> None:
    t0 = time.perf_counter()
    first_tok_t = persona_applied = None
    ev = None
    with _post(base + "/ask/stream", payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
                if ev == "meta":
                    _print_sources(data.get("sources", []),
                                   data.get("top_score"), data.get("min_score"))
                    if data.get("rewritten_question"):
                        print(_dim(f"(rewritten → {data['rewritten_question']})") + "\n")
                    print("khoji ›  ", end="", flush=True)
                elif ev == "token":
                    if first_tok_t is None:
                        first_tok_t = time.perf_counter() - t0
                    print(data["text"], end="", flush=True)
                elif ev == "done":
                    m = data.get("meta", {})
                    persona_applied = data.get("persona_applied", False)
                    print("\n")
                    _footer(persona_applied, first_tok_t, time.perf_counter() - t0,
                            m.get("output_tokens"), m.get("tok_per_s"), m.get("load_ms"))


def _run_blocking(base: str, payload: dict) -> None:
    t0 = time.perf_counter()
    with _post(base + "/ask", payload) as r:
        resp = json.loads(r.read())
    total = time.perf_counter() - t0
    _print_sources(resp.get("sources", []), resp.get("top_score"), resp.get("min_score"))
    if resp.get("rewritten_question"):
        print(_dim(f"(rewritten → {resp['rewritten_question']})") + "\n")
    print("khoji ›\n" + resp["answer"] + "\n")
    tim = resp["timings"]
    # blocking /ask doesn't stream, so no TTFT and no token metrics in the contract —
    # show the server-side timing breakdown instead.
    print(_dim(
        f"persona {'on' if resp.get('persona_applied') else 'off'} · wall {total:.1f}s · "
        f"rewrite {tim['rewrite_ms']/1000:.1f}s · retrieve {tim['retrieve_ms']/1000:.1f}s · "
        f"generate {tim['generate_ms']/1000:.1f}s"))


def _footer(persona_applied, ttft_s, total_s, out_tok, tok_s, load_ms) -> None:
    ttft = f"{ttft_s:.1f}s" if ttft_s is not None else "-"
    load = f"{(load_ms or 0) / 1000:.1f}s"
    print(_dim(f"persona {'on' if persona_applied else 'off'} · ttft {ttft} · "
               f"total {total_s:.1f}s · {out_tok} tok @ {tok_s} tok/s · load {load}"))


def _print_sources(sources: list[dict], top_score=None, min_score=None) -> None:
    if not sources:
        if top_score is not None:
            floor = f"the {min_score} floor" if min_score is not None else "the relevance floor"
            print(_dim(f"sources: nothing cleared {floor} (best match {top_score}) — refusing") + "\n")
        else:
            print(_dim("sources: (none retrieved — is a corpus ingested?)") + "\n")
        return
    head = f"sources (top {top_score}):" if top_score is not None else "sources:"
    lines = [head]
    for i, s in enumerate(sources, 1):
        lines.append(f"  [{i}] {s.get('subject')} · {s.get('chapter')} · "
                     f"p{s.get('page')} · {s.get('score')}")
    print(_dim("\n".join(lines)) + "\n")


def _ask_once(base: str, question: str, session: str | None, persona: bool, stream: bool) -> None:
    payload = {"question": question, "session_id": session, "persona": persona}
    if stream:
        _run_stream(base, payload)
    else:
        _run_blocking(base, payload)


def _chat_loop(base: str, session: str | None, persona: bool, stream: bool) -> None:
    session = session or f"chat-{int(time.time())}"
    print(_dim("─" * 60))
    print("Khoji — offline NCERT tutor")
    print(_dim(f"session {session}  ·  persona {'on' if persona else 'off'}"))
    print(_dim("type a question  ·  /persona toggles voice  ·  /new resets session  "
               "·  exit/quit to leave"))
    print(_dim("─" * 60) + "\n")

    while True:
        try:
            q = input("you ›  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + _dim("bye — khoji signing off"))
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            print(_dim("bye — khoji signing off"))
            break
        if q.lower() == "/persona":
            persona = not persona
            print(_dim(f"persona {'on' if persona else 'off'}") + "\n")
            continue
        if q.lower() == "/new":
            session = f"chat-{int(time.time())}"
            print(_dim(f"fresh session {session}") + "\n")
            continue

        try:
            _ask_once(base, q, session, persona, stream)
        except urllib.error.URLError as e:
            print(_dim(f"! could not reach the server at {base} — is uvicorn running? ({e})"))
        print()


def main() -> None:
    # Windows consoles default to cp1252 and crash on Greek/math chars (α, °, ×)
    # that science answers are full of. Force UTF-8, never crash on output.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Ask Khoji from the terminal.")
    ap.add_argument("question", nargs="?", default=None,
                    help="a one-shot question; omit to start an interactive chat loop")
    ap.add_argument("--session", default=None, help="session id (enables follow-up rewrite)")
    ap.add_argument("--no-stream", action="store_true", help="use blocking /ask")
    ap.add_argument("--persona", action="store_true", help="opt into the Khoji Bhai re-voice")
    ap.add_argument("--no-color", action="store_true", help="disable dim styling")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global _COLOR
    _COLOR = (not args.no_color) and sys.stdout.isatty()
    if _COLOR and sys.platform == "win32":
        # Best-effort: enable ANSI (VT) processing on modern Windows consoles.
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            _COLOR = False

    base = f"http://localhost:{args.port}"
    stream = not args.no_stream

    if args.question is None:
        _chat_loop(base, args.session, args.persona, stream)
        return

    try:
        _ask_once(base, args.question, args.session, args.persona, stream)
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach the server at {base} — is uvicorn running? ({e})")


if __name__ == "__main__":
    main()
