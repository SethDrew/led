#!/usr/bin/env python3
"""
Audio segment tool — launches the browser-based explorer.

Usage:
    python segment.py                    # Launch web UI
    python segment.py --port 8080        # Custom port
    python segment.py --no-browser       # Don't auto-open browser
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Audio segment explorer')
    parser.add_argument('--port', type=int, default=0, help='Server port (0=auto)')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (default localhost)')
    parser.add_argument('--no-browser', action='store_true', help='Skip opening browser')
    args = parser.parse_args()

    from web_viewer import run_server
    run_server(port=args.port, host=args.host, no_browser=args.no_browser)


if __name__ == '__main__':
    main()
