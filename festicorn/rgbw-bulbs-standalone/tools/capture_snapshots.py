#!/usr/bin/env python3
"""
Capture LED frame data from the RGBW bulbs device and render ledogram snapshots.

Usage:
    python3 capture_snapshots.py --ip 192.168.4.25 --effect bloom --seconds 5
    python3 capture_snapshots.py --ip 192.168.4.25 --all --seconds 5
"""

import argparse
import socket
import struct
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

# Find preview_lib: led/tools/preview_lib.py (three dirs up from this script)
_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / 'tools'))

try:
    import numpy as np
    from PIL import Image
    from preview_lib import render_ledogram, apply_gamma
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install numpy pillow")
    sys.exit(1)

CAPTURE_PORT = 4555
EFFECTS = ['bloom', 'fire', 'leaf_wind']


def post(ip: str, path: str, params: dict) -> str:
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(
        f'http://{ip}{path}',
        data=data,
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode()


def capture(ip: str, effect: str, seconds: int, num_leds: int = 100) -> np.ndarray:
    """Switch device to effect, open UDP, trigger capture, collect frames."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', CAPTURE_PORT))
    sock.settimeout(2.0)

    packet_size = 4 + num_leds * 4  # 4-byte frame index + RGBW bytes
    frames_by_idx: dict[int, np.ndarray] = {}

    print(f"  Triggering capture: effect={effect}, seconds={seconds} ...", flush=True)
    post(ip, '/capture', {'effect': effect, 'seconds': str(seconds)})

    deadline = time.monotonic() + seconds + 2.0  # +2s grace
    while time.monotonic() < deadline:
        try:
            pkt, _ = sock.recvfrom(packet_size + 16)
        except socket.timeout:
            continue
        if len(pkt) < packet_size:
            continue
        idx = struct.unpack_from('<I', pkt, 0)[0]
        rgbw = np.frombuffer(pkt, dtype=np.uint8, count=num_leds * 4, offset=4)
        rgbw = rgbw.reshape(num_leds, 4)
        frames_by_idx[idx] = rgbw

    sock.close()

    if not frames_by_idx:
        raise RuntimeError("No frames received — check device IP and UDP connectivity")

    max_idx = max(frames_by_idx)
    print(f"  Received {len(frames_by_idx)}/{max_idx+1} frames "
          f"({len(frames_by_idx)/(max_idx+1)*100:.0f}% received)", flush=True)

    # Build dense array (drop W, keep R G B for ledogram)
    rgb_frames = np.zeros((max_idx + 1, num_leds, 3), dtype=np.uint8)
    for idx, rgbw in frames_by_idx.items():
        rgb_frames[idx, :, 0] = rgbw[:, 0]  # R
        rgb_frames[idx, :, 1] = rgbw[:, 1]  # G
        rgb_frames[idx, :, 2] = rgbw[:, 2]  # B

    return rgb_frames


def render_snapshot(frames: np.ndarray, fps: float, out_path: Path) -> None:
    corrected = apply_gamma(frames)
    img = render_ledogram(corrected, fps=fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    print(f"  Saved {out_path} ({img.width}x{img.height})", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Capture LED snapshots from RGBW bulbs device')
    parser.add_argument('--ip',      default='192.168.4.25', help='Device IP')
    parser.add_argument('--effect',  choices=EFFECTS,        help='Single effect to capture')
    parser.add_argument('--all',     action='store_true',    help='Capture all effects')
    parser.add_argument('--seconds', type=int, default=5,    help='Capture duration per effect')
    parser.add_argument('--fps',     type=float, default=244.0, help='Device FPS (for ledogram time axis)')
    parser.add_argument('--leds',    type=int, default=100,  help='Number of LEDs')
    parser.add_argument('--out',     default=None,           help='Output path (single effect only)')
    args = parser.parse_args()

    if not args.effect and not args.all:
        parser.error('Specify --effect NAME or --all')

    effects = EFFECTS if args.all else [args.effect]

    # Default output directory: data/snapshots/ relative to this script's parent
    data_snapshots = Path(__file__).resolve().parents[1] / 'data' / 'snapshots'

    for effect in effects:
        print(f"\n[{effect}]", flush=True)
        if args.out and len(effects) == 1:
            out_path = Path(args.out)
        else:
            out_path = data_snapshots / f'{effect}.png'

        try:
            frames = capture(args.ip, effect, args.seconds, args.leds)
            render_snapshot(frames, args.fps, out_path)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            if len(effects) == 1:
                sys.exit(1)

        # Brief settle time between effects
        if effect != effects[-1]:
            time.sleep(1.0)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
