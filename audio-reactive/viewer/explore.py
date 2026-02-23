#!/usr/bin/env python3
"""
Audio Explorer — launches the browser-based analysis viewer.

Usage:
    python explore.py                    # Launch web UI
    python explore.py --port 8080        # Custom port
    python explore.py --no-browser       # Don't auto-open browser
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Audio Explorer')
    parser.add_argument('--port', type=int, default=0, help='Server port (0=auto)')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (default localhost)')
    parser.add_argument('--no-browser', action='store_true', help='Skip opening browser')
    args = parser.parse_args()

    from web.server import run_server
    run_server(port=args.port, host=args.host, no_browser=args.no_browser)


if __name__ == '__main__':
    main()
