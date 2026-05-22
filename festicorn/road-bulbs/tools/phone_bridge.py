#!/usr/bin/env python3
"""
phone_bridge.py — Sensor Logger HTTP push → serial SensorPacket frames.

Listens for JSON POSTs from the "Sensor Logger" phone app, extracts the most
recent accel / gyro / (optional) microphone readings from each push, and
sends framed SensorPackets to the road-bulbs ESP32 at 25Hz.

Default audio source is the Mac built-in mic via sounddevice (same RMS
pipeline as festicorn/tools/mic_bridge.py). Use --phone-mic to use the
Sensor Logger microphone payload instead.

Sensor Logger payload shape (POSTed as JSON):
    {
      "payload": [
        {"name": "totalacceleration", "values": {"x":.., "y":.., "z":..},
         "time": <ns>},
        {"name": "gyroscope",         "values": {"x":.., "y":.., "z":..},
         "time": <ns>},
        {"name": "microphone",        "values": {...},                  ...},
        ...
      ]
    }
There are typically many readings per sensor per push (~1s batches). We use
the latest-by-time entry for each sensor.

Frame: [0xA5][0x5A][LEN][PAYLOAD][XOR8]  (XOR8 = LEN ^ all payload bytes)

SensorPacket (15 bytes, little-endian):
    int16 ax, ay, az    (raw MPU-6050, 16384 = 1g)
    int16 gx, gy, gz    (raw MPU-6050, /131 = deg/s)
    uint16 rawRms       (audio RMS, 0..65535)
    uint8 micEnabled    (1 if any audio source active, else 0)

Usage:
    phone_bridge.py --port /dev/cu.usbserial-8
    phone_bridge.py --listen-port 8765 --phone-mic --mic-units dbfs
"""

import argparse
import json
import math
import signal
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import serial
import sounddevice as sd

FRAME_MAGIC = b"\xA5\x5A"
SENSOR_STRUCT = struct.Struct("<hhhhhh HB")  # 15 bytes
PACKET_HZ = 25.0

ACCEL_SCALE = 16384.0 / 9.81
GYRO_SCALE = 131.0 * 180.0 / math.pi

SAMPLE_RATE = 16000
RMS_SCALE_DEFAULT = 1_500_000.0  # float[-1,1] RMS → uint16


# ── Mac mic capture (verbatim from mic_bridge.MicRMS) ────────────

class MicRMS:
    def __init__(self, scale: float = RMS_SCALE_DEFAULT):
        self.scale = scale
        self.window = np.zeros(SAMPLE_RATE * 80 // 1000, dtype=np.float32)
        self.idx = 0
        self.lock = threading.Lock()
        self.captured = 0

    def callback(self, indata, frames, t, status):
        x = indata[:, 0]
        n = len(x)
        with self.lock:
            w = self.window
            i = self.idx
            if n >= len(w):
                w[:] = x[-len(w):]
                self.idx = 0
            else:
                end = i + n
                if end <= len(w):
                    w[i:end] = x
                else:
                    first = len(w) - i
                    w[i:] = x[:first]
                    w[:n - first] = x[first:]
                self.idx = end % len(w)
            self.captured += n

    def rms_u16(self) -> int:
        with self.lock:
            w = self.window
            r = float(np.sqrt(np.mean(w * w)))
        return int(min(65535, max(0, r * self.scale)))


# ── Shared sensor state filled by HTTP push handler ──────────────

class SensorState:
    """Latest reading per sensor, thread-safe. Values stored in source units
    (m/s², rad/s, raw mic payload dict)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.accel = None         # (x, y, z) m/s²
        self.gyro = None          # (x, y, z) rad/s
        self.mic_value = None     # raw dict from Sensor Logger microphone entry
        self.last_post_t = 0.0
        self.posts = 0
        self.seen_sensors = set()

    def update_from_payload(self, payload):
        """payload: list of {name, values, time} dicts."""
        # Group by name, keep the latest-by-time entry per sensor.
        latest = {}
        for entry in payload:
            name = entry.get("name")
            t = entry.get("time", 0)
            if not name:
                continue
            cur = latest.get(name)
            if cur is None or t > cur.get("time", 0):
                latest[name] = entry

        with self.lock:
            self.last_post_t = time.monotonic()
            self.posts += 1
            for name, entry in latest.items():
                self.seen_sensors.add(name)
                vals = entry.get("values", {}) or {}
                if name == "totalacceleration":
                    self.accel = (float(vals.get("x", 0.0)),
                                  float(vals.get("y", 0.0)),
                                  float(vals.get("z", 0.0)))
                elif name == "gyroscope":
                    self.gyro = (float(vals.get("x", 0.0)),
                                 float(vals.get("y", 0.0)),
                                 float(vals.get("z", 0.0)))
                elif name == "microphone":
                    self.mic_value = vals

    def snapshot(self):
        with self.lock:
            return (self.accel, self.gyro, self.mic_value,
                    self.last_post_t, self.posts, set(self.seen_sensors))


# ── HTTP push server ─────────────────────────────────────────────

def make_handler(state: SensorState, verbose: bool):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if verbose:
                sys.stderr.write("[http] " + (fmt % args) + "\n")

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return None
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None

        def do_POST(self):
            data = self._read_json()
            if isinstance(data, dict):
                payload = data.get("payload")
                if isinstance(payload, list):
                    state.update_from_payload(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_GET(self):
            # Helpful sanity probe: hitting it in a browser confirms reachability.
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"road-bulbs bridge: POST Sensor Logger JSON here\n")

    return Handler


# ── Phone mic conversion ─────────────────────────────────────────

def phone_mic_to_rms_u16(vals: dict, units: str) -> int:
    """Convert Sensor Logger microphone values dict to a uint16 RMS."""
    if not isinstance(vals, dict):
        return 0
    # Sensor Logger's microphone entry layout varies — handle a few:
    #   {"dBFS": -23.4}     → log magnitude in dBFS (negative)
    #   {"amplitude": 0.04} → linear 0..1
    #   {"rms": 0.012}      → linear 0..1
    if units == "dbfs":
        db = vals.get("dBFS")
        if db is None:
            return 0
        # dBFS is ≤ 0. Map -60 dBFS → 0, 0 dBFS → 65535.
        x = max(0.0, min(1.0, (float(db) + 60.0) / 60.0))
        return int(x * 65535)
    if units == "linear":
        v = vals.get("amplitude")
        if v is None:
            v = vals.get("rms", 0.0)
        return max(0, min(65535, int(float(v) * 50000.0)))
    return 0


# ── Frame ────────────────────────────────────────────────────────

def make_frame(payload: bytes) -> bytes:
    length = len(payload)
    xor = length
    for b in payload:
        xor ^= b
    return FRAME_MAGIC + bytes([length]) + payload + bytes([xor & 0xFF])


def clamp_i16(v: float) -> int:
    return max(-32768, min(32767, int(v)))


# ── Main ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/cu.usbserial-8",
                   help="ESP32 serial device")
    p.add_argument("--baud", type=int, default=460800, help="Serial baud rate")
    p.add_argument("--listen-host", default="0.0.0.0",
                   help="Interface to bind HTTP server")
    p.add_argument("--listen-port", type=int, default=8765,
                   help="HTTP port for Sensor Logger pushes (default 8765)")
    p.add_argument("--mac-mic", action="store_true",
                   help="Use Mac built-in mic instead of Sensor Logger microphone")
    p.add_argument("--mic-units", choices=("dbfs", "linear"), default="dbfs",
                   help="Interpretation of phone microphone values (default: dbfs)")
    p.add_argument("--no-mic", action="store_true",
                   help="Disable all audio (rawRms=0, micEnabled=0)")
    p.add_argument("--mic-scale", type=float, default=RMS_SCALE_DEFAULT,
                   help=f"Mac mic RMS scale (default: {RMS_SCALE_DEFAULT})")
    p.add_argument("--mic-device", default=None,
                   help="sounddevice input device name/index")
    p.add_argument("--http-verbose", action="store_true",
                   help="Log every HTTP request")
    args = p.parse_args()

    state = SensorState()

    # ── Audio source ─────────────────────────────────────────────
    mic = None
    audio_stream = None
    use_mac_mic = not args.no_mic and args.mac_mic
    if use_mac_mic:
        try:
            mic = MicRMS(args.mic_scale)
            audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                device=args.mic_device, callback=mic.callback, blocksize=512)
            audio_stream.start()
        except Exception as e:
            print(f"WARN: Mac mic init failed ({e}); audio disabled.",
                  file=sys.stderr)
            mic = None
            audio_stream = None
            use_mac_mic = False

    print(f"Serial:       {args.port} @ {args.baud}")
    print(f"Listening on  http://{args.listen_host}:{args.listen_port} for Sensor Logger push...")
    if use_mac_mic and mic is not None:
        print(f"Mic:          Mac (sounddevice, scale={args.mic_scale:.0f})")
    elif not args.no_mic:
        print(f"Mic:          Sensor Logger payload (units={args.mic_units})")
    else:
        print("Mic:          disabled")

    # ── HTTP server in a thread ─────────────────────────────────
    httpd = ThreadingHTTPServer((args.listen_host, args.listen_port),
                                make_handler(state, args.http_verbose))
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    # ── Serial ───────────────────────────────────────────────────
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(0.5)
    ser.reset_input_buffer()

    running = [True]

    def stop(signum, frame):
        running[0] = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    period = 1.0 / PACKET_HZ
    next_tick = time.monotonic()
    sent = 0
    last_stat = time.monotonic()
    last_posts = 0
    announced_sensors = set()

    while running[0]:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
        next_tick += period
        if next_tick < time.monotonic() - period:
            next_tick = time.monotonic() + period

        accel, gyro, mic_value, last_post_t, posts, seen = state.snapshot()

        # Announce newly-seen sensors once.
        new = seen - announced_sensors
        if new:
            for n in sorted(new):
                print(f"[detected] {n}")
            announced_sensors |= new

        if accel is None:
            ax = ay = az = 0
        else:
            ax = clamp_i16(accel[0] * ACCEL_SCALE)
            ay = clamp_i16(accel[1] * ACCEL_SCALE)
            az = clamp_i16(accel[2] * ACCEL_SCALE)
        if gyro is None:
            gx = gy = gz = 0
        else:
            gx = clamp_i16(gyro[0] * GYRO_SCALE)
            gy = clamp_i16(gyro[1] * GYRO_SCALE)
            gz = clamp_i16(gyro[2] * GYRO_SCALE)

        if args.no_mic:
            rms = 0
            mic_on = 0
        elif use_mac_mic and mic is not None:
            rms = mic.rms_u16()
            mic_on = 1
        elif not use_mac_mic and mic_value is not None:
            rms = phone_mic_to_rms_u16(mic_value, args.mic_units)
            mic_on = 1
        else:
            rms = 0
            mic_on = 0

        payload = SENSOR_STRUCT.pack(ax, ay, az, gx, gy, gz, rms, mic_on)
        try:
            ser.write(make_frame(payload))
            sent += 1
        except serial.SerialException as e:
            print(f"[serial] {e}", file=sys.stderr)
            break

        if ser.in_waiting:
            try:
                line = ser.readline().decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"[esp] {line}")
            except Exception:
                pass

        if now - last_stat >= 1.0:
            posts_per_s = posts - last_posts
            last_posts = posts
            stale = (now - last_post_t) if last_post_t else float("inf")
            print(f"sent={sent}/s posts={posts_per_s}/s stale={stale:5.2f}s "
                  f"a=({ax},{ay},{az}) g=({gx},{gy},{gz}) rms={rms}")
            sent = 0
            last_stat = now

    print("Shutting down.")
    try:
        httpd.shutdown()
    except Exception:
        pass
    if audio_stream is not None:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass
    ser.close()


if __name__ == "__main__":
    main()
