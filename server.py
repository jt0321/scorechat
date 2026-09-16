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
import urllib.parse
from http.server import SimpleHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv

# Load env variables before importing local modules
load_dotenv()

from db.store import list_works, get_work_mei

PORT = 8000
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


class ScoreChatHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve static files from the frontend directory
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)

        # Which chat providers this deployment can actually use. The picker in
        # the client is built from this: a name and whether its credentials are
        # present, never the credentials themselves.
        if parsed_url.path == "/api/providers":
            try:
                from pipeline.providers import chat_provider_options
                self._send_json(200, {"providers": chat_provider_options()})
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
            question = query_params.get("question", [""])[0]
            if not question:
                self._send_json(400, {"error": "Missing question parameter"})
                return
            provider = query_params.get("provider", [""])[0] or None
            model = query_params.get("model", [""])[0] or None
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
                self._send_json(200, answer(question, model=model, provider=provider))
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        # Fallback to default static file serving
        return super().do_GET()


def main():
    print(f"=========================================")
    print(f"ScoreChat Backend running on port {PORT}")
    print(f"Open http://localhost:{PORT} in your browser")
    print(f"=========================================")
    server = HTTPServer(("0.0.0.0", PORT), ScoreChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
