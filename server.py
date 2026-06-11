"""
server.py
---------
Simple Python http.server-based backend to serve the frontend and expose
the ScoreChat RAG API at /api/chat.

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

from pipeline.chat import chat

PORT = 8000
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


class ScoreChatHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve static files from the frontend directory
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)

        # RAG Chat API Endpoint
        if parsed_url.path == "/api/chat":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            query = query_params.get("query", [""])[0]

            if not query:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing query parameter"}).encode("utf-8"))
                return

            try:
                # Call RAG pipeline
                result = chat(query)
                response_data = json.dumps(result)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                # Enable CORS
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(response_data.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
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
