#!/usr/bin/env python3
"""
festicorn/web/server.py — standalone web UI for the mic_bridge recorder.

Serves static/{index.html, app.js, style.css} and delegates /api/* to
RecorderHandler from festicorn.tools.mic_bridge (same handler the viewer
used in the prior topology).

Run:   .venv/bin/python festicorn/web/server.py [--bridge-port /dev/cu.usbmodemXXX] [--http 8765]
Open:  http://localhost:8765/
"""

from __future__ import annotations
import argparse, os, sys, threading, mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import mic_bridge  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
HTTP_PORT_DEFAULT = 8765


class FesticornHandler(mic_bridge.RecorderHandler):
    """Inherits /api/* handling from RecorderHandler; adds static file
    serving for / and /static/*."""

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            return super().do_GET()
        if path == "/" or path == "/index.html":
            return self._serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            full = os.path.normpath(os.path.join(STATIC_DIR, rel))
            if not full.startswith(STATIC_DIR):
                self._json(403, {"error": "forbidden"}); return
            if not os.path.isfile(full):
                self._json(404, {"error": "not found"}); return
            ctype, _ = mimetypes.guess_type(full)
            return self._serve_file(full, ctype or "application/octet-stream")
        return super().do_GET()

    def _serve_file(self, path: str, ctype: str):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-port", help="espnow-bridge serial port (omit for mic-only)")
    ap.add_argument("--http", type=int, default=HTTP_PORT_DEFAULT,
                    help=f"HTTP port (default {HTTP_PORT_DEFAULT})")
    args = ap.parse_args()

    bridge_port = mic_bridge.discover_bridge_port(args.bridge_port)
    print(f"bridge : {bridge_port or '(none — mic-only)'}")
    rec = mic_bridge.Recorder(bridge_port, mic_bridge.RMS_SCALE_DEFAULT)
    rec.open()
    FesticornHandler.recorder = rec

    httpd = HTTPServer(("0.0.0.0", args.http), FesticornHandler)
    print(f"http   : http://localhost:{args.http}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rec.close()


if __name__ == "__main__":
    main()
