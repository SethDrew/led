"""Smoke test for rgbw-bulbs-standalone web UI at http://192.168.4.25

macOS TCC / Local Network note
-------------------------------
Playwright's chromium.launch() — headless or not — inherits the terminal
session's sandbox and cannot reach LAN addresses even if System Settings shows
the permission as ON. The grant only applies when Chrome for Testing is launched
via LaunchServices (i.e. `open -a`), which gives it its own sandbox context.

The working approach: launch Chrome for Testing with `open -a` and a dedicated
user-data-dir + remote-debugging-port, then drive it via raw CDP over websockets.
This script does that automatically. It is idempotent: if Chrome for Testing is
already listening on the debug port it reuses that instance.

To pre-launch manually (e.g. after a reboot):
    open -a ~/Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/"Google Chrome for Testing.app" \\
        --args --remote-debugging-port=9223 --no-first-run --user-data-dir=/tmp/cft-qa
"""
import asyncio
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

try:
    import websockets
except ImportError:
    raise SystemExit("Run: .venv/bin/pip install websockets")

DEVICE_URL = "http://192.168.4.25"
DEBUG_PORT = 9223
CDP_HTTP = f"http://localhost:{DEBUG_PORT}"
SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
CHROMIUM_APP = (
    Path.home()
    / "Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64"
    / "Google Chrome for Testing.app"
)
USER_DATA_DIR = "/tmp/cft-qa"


def ensure_browser():
    """Launch Chrome for Testing via open -a if not already on DEBUG_PORT."""
    try:
        urllib.request.urlopen(f"{CDP_HTTP}/json/version", timeout=2)
        print(f"Chrome for Testing already on port {DEBUG_PORT}, reusing.")
        return
    except Exception:
        pass

    print(f"Launching Chrome for Testing via open -a on port {DEBUG_PORT} ...")
    subprocess.Popen([
        "open", "-a", str(CHROMIUM_APP),
        "--args",
        f"--remote-debugging-port={DEBUG_PORT}",
        "--no-first-run",
        f"--user-data-dir={USER_DATA_DIR}",
    ])
    # Wait for CDP to become available
    for _ in range(20):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"{CDP_HTTP}/json/version", timeout=1)
            print("CDP ready.")
            return
        except Exception:
            pass
    raise RuntimeError(f"Chrome for Testing did not open CDP on port {DEBUG_PORT} within 10s")


def open_tab():
    """Open a new blank tab and return its websocket debugger URL."""
    targets = json.loads(urllib.request.urlopen(f"{CDP_HTTP}/json/list", timeout=3).read())
    pages = [t for t in targets if t["type"] == "page"]
    if pages:
        return pages[0]["webSocketDebuggerUrl"]
    req = urllib.request.Request(f"{CDP_HTTP}/json/new?about:blank", method="PUT")
    target = json.loads(urllib.request.urlopen(req, timeout=3).read())
    return target["webSocketDebuggerUrl"]


async def run():
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    ws_url = open_tab()
    print(f"CDP target: {ws_url[:70]}")

    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        seq = 0

        async def cdp(method, params=None):
            nonlocal seq
            seq += 1
            mid = seq
            await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(f"CDP {method}: {msg['error']}")
                    return msg.get("result", {})

        async def screenshot(name):
            r = await cdp("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
            path = SCREENSHOTS_DIR / name
            path.write_bytes(base64.b64decode(r["data"]))
            print(f"Screenshot saved: {path}")
            return path

        await cdp("Page.enable")
        await cdp("Runtime.enable")
        await cdp("Emulation.setDeviceMetricsOverride", {
            "width": 480, "height": 800, "deviceScaleFactor": 1, "mobile": False,
        })

        print(f"Navigating to {DEVICE_URL} ...")
        nav = await cdp("Page.navigate", {"url": DEVICE_URL})
        if nav.get("errorText"):
            raise RuntimeError(f"Navigation failed: {nav['errorText']}")

        # Wait for effect cards (JS fetches /effects and builds them dynamically)
        await asyncio.sleep(1.5)
        for _ in range(30):
            r = await cdp("Runtime.evaluate", {
                "expression": "document.querySelectorAll('.effect-card').length",
                "returnByValue": True,
            })
            if r.get("result", {}).get("value", 0) > 0:
                break
            await asyncio.sleep(0.3)
        else:
            raise RuntimeError("Effect cards never appeared — /effects endpoint unreachable?")

        card_count = r["result"]["value"]
        print(f"Cards loaded: {card_count}")

        # 1. Collapsed state
        await screenshot("collapsed.png")

        # Expand picker
        await cdp("Runtime.evaluate", {
            "expression": "document.getElementById('effect-header').click()",
        })
        await asyncio.sleep(0.4)

        # 2. Expanded state
        await screenshot("expanded.png")

        # Click Leaf Wind
        result = (await cdp("Runtime.evaluate", {
            "expression": "(function(){ var c=document.getElementById('card-leaf_wind'); if(!c) return 'not found'; c.click(); return 'clicked'; })()",
            "returnByValue": True,
        }))["result"]["value"]
        if result != "clicked":
            raise RuntimeError(f"Leaf Wind card: {result}")
        print(f"Leaf Wind card: {result}")
        await asyncio.sleep(1.5)

        # Verify /status
        status_raw = (await cdp("Runtime.evaluate", {
            "expression": "(async()=>{ const r=await fetch('/status'); return JSON.stringify(await r.json()); })()",
            "awaitPromise": True,
            "returnByValue": True,
        }))["result"]["value"]
        status = json.loads(status_raw)
        print(f"Status: {json.dumps(status)}")
        effect = status.get("effect", "")
        if effect == "leaf_wind":
            print("PASS: effect == 'leaf_wind'")
        else:
            print(f"FAIL: expected 'leaf_wind', got '{effect}'")

        # 3. Active state
        await screenshot("leaf_wind_active.png")

    print("Done.")


def main():
    ensure_browser()
    asyncio.run(run())


if __name__ == "__main__":
    main()
