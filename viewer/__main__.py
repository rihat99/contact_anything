"""Run the contact dataset viewer with ``python -m viewer``."""
from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Browse contact datasets in a light web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser window")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"Contact dataset viewer: {url}")
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("viewer.app:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
