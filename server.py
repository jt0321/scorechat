"""
server.py
---------
Simple Python http.server-based backend to serve the frontend and expose
the ScoreChat analysis API at /api/ask.

Usage:
    python server.py
"""

import os
import json
import threading
import time
import urllib.parse
from collections import defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from dotenv import load_dotenv

# Load env variables before importing local modules
load_dotenv()

from db.store import list_works, get_work_mei

# A host assigns the port; 8000 is only the local default.
PORT = int(os.environ.get("PORT", "8000"))
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

# Which providers this deployment offers, as a comma-separated list of names.
# Unset means every provider whose credentials are present — right locally, and
# wrong for a public URL, where the deployment's key answers for whoever asks.
ALLOWED_PROVIDERS = [
    name.strip() for name in os.environ.get("ALLOWED_CHAT_PROVIDERS", "").split(",")
    if name.strip()
]

# Per-IP budget for /api/ask, the only endpoint that spends money. 0 disables it.
ASK_RATE_LIMIT  = int(os.environ.get("ASK_RATE_LIMIT", "20"))
ASK_RATE_WINDOW = int(os.environ.get("ASK_RATE_WINDOW", "3600"))
# Forwarded client IPs are only believable behind a proxy that sets them; taken
# at face value elsewhere they are a header the caller writes, so every request
# can claim a fresh address and the limit above counts nothing.
TRUST_PROXY = os.environ.get("TRUST_PROXY", "").lower() in ("1", "true", "yes")

_ask_history: dict[str, deque] = defaultdict(deque)
_ask_lock = threading.Lock()


def _rate_limited(client_ip: str) -> int | None:
    """Seconds until this IP may ask again, or None if it may ask now."""
    if ASK_RATE_LIMIT <= 0:
        return None
    now = time.monotonic()
    with _ask_lock:
        history = _ask_history[client_ip]
        while history and now - history[0] > ASK_RATE_WINDOW:
            history.popleft()
        if len(history) >= ASK_RATE_LIMIT:
            return int(ASK_RATE_WINDOW - (now - history[0])) + 1
        history.append(now)
        # Addresses that stopped asking would otherwise accumulate for the life
        # of the process; an empty history is the same as an absent one.
        if len(_ask_history) > 10_000:
            for ip in [ip for ip, h in _ask_history.items() if not h]:
                del _ask_history[ip]
    return None


# Big enough for a question and its recent history, which the server clips anyway.
MAX_ASK_BODY_BYTES = 64 * 1024


class ScoreChatHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve static files from the frontend directory
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def _client_ip(self) -> str:
        if TRUST_PROXY:
            forwarded = (self.headers.get("Fly-Client-IP")
                         or self.headers.get("X-Forwarded-For", "").split(",")[0])
            if forwarded.strip():
                return forwarded.strip()
        return self.client_address[0]

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _ask(self, question, provider, model, history=None, selection=None) -> None:
        """Answer one question. GET carries a lone question; POST adds the
        conversation so far, which a follow-up or the reply to a clarifying
        question cannot be understood without, and the bars selected in the
        score viewer, which "these bars" cannot."""
        if not isinstance(question, str) or not question.strip():
            self._send_json(400, {"error": "Missing question parameter"})
            return
        if provider and ALLOWED_PROVIDERS and provider not in ALLOWED_PROVIDERS:
            self._send_json(403, {"error": f"Provider '{provider}' is not enabled "
                                           f"on this deployment."})
            return
        retry_after = _rate_limited(self._client_ip())
        if retry_after is not None:
            self._send_json(429, {"error": f"Rate limit reached "
                                           f"({ASK_RATE_LIMIT} questions per "
                                           f"{ASK_RATE_WINDOW // 60} minutes). "
                                           f"Try again in {retry_after}s."})
            return
        try:
            from pipeline.providers import CHAT_PROVIDERS
            from pipeline.tools import answer
            if provider and provider not in CHAT_PROVIDERS:
                # Named rather than silently ignored: falling back to the
                # env default would answer with a model the user did not
                # pick and give no sign of it.
                self._send_json(400, {"error": f"Unknown provider '{provider}'. "
                                               f"Supported: {', '.join(CHAT_PROVIDERS)}."})
                return
            self._send_json(200, answer(question.strip(), model=model, provider=provider,
                                        history=history, selection=selection))
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_POST(self) -> None:
        if urllib.parse.urlparse(self.path).path != "/api/ask":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_ASK_BODY_BYTES:
            self._send_json(413 if length > 0 else 400,
                            {"error": "Request body missing or too large"})
            return
        try:
            body = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request body is not JSON"})
            return
        if not isinstance(body, dict):
            self._send_json(400, {"error": "Request body must be a JSON object"})
            return
        text_or_none = lambda value: value if isinstance(value, str) and value else None
        self._ask(body.get("question"), text_or_none(body.get("provider")),
                  text_or_none(body.get("model")), body.get("history"),
                  body.get("selection"))

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)

        # Which chat providers this deployment can actually use. The picker in
        # the client is built from this: a name and whether its credentials are
        # present, never the credentials themselves.
        if parsed_url.path == "/api/providers":
            try:
                from pipeline.providers import chat_provider_options
                options = chat_provider_options()
                if ALLOWED_PROVIDERS:
                    options = [o for o in options if o["id"] in ALLOWED_PROVIDERS]
                    # CHAT_PROVIDER may be one this deployment does not offer, and
                    # a picker with nothing selected offers no way in.
                    if options and not any(o["selected"] for o in options):
                        options[0]["selected"] = True
                self._send_json(200, {"providers": options})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        # List all ingested works (for the sidebar work picker)
        if parsed_url.path == "/api/works":
            try:
                self._send_json(200, {"works": list_works()})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        # Full MEI for a work, so the frontend can load the whole score
        # into Verovio and support real pagination/navigation instead of
        # rendering an isolated per-segment slice.
        if parsed_url.path == "/api/score":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            work_id_raw = query_params.get("work_id", [""])[0]
            if not work_id_raw.isdigit():
                self._send_json(400, {"error": "Missing or invalid work_id parameter"})
                return

            try:
                mei = get_work_mei(int(work_id_raw))
                if mei is None:
                    self._send_json(404, {"error": "No MEI asset found for this work"})
                    return
                body = mei.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        # Tool-calling analysis endpoint, and what the HTML client asks. It
        # returns prose plus the trace of stored analysis the answer was built
        # from; the client renders that trace as the citation, and reads the
        # bar ranges out of the calls' arguments to open the score at them.
        if parsed_url.path == "/api/ask":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            self._ask(query_params.get("question", [""])[0],
                      query_params.get("provider", [""])[0] or None,
                      query_params.get("model", [""])[0] or None)
            return

        # Fallback to default static file serving
        return super().do_GET()


def main():
    print(f"=========================================")
    print(f"ScoreChat Backend running on port {PORT}")
    print(f"Open http://localhost:{PORT} in your browser")
    if ALLOWED_PROVIDERS:
        print(f"Providers: {', '.join(ALLOWED_PROVIDERS)}")
    if ASK_RATE_LIMIT > 0:
        print(f"/api/ask limit: {ASK_RATE_LIMIT} per {ASK_RATE_WINDOW}s per IP")
    print(f"=========================================")
    # Threaded: an answer is several seconds of tool calls and model round-trips,
    # and on a single-threaded server that blocks every other request — including
    # the page itself and the score it renders.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), ScoreChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
