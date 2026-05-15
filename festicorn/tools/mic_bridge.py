#!/usr/bin/env python3
"""
mic_bridge.py — laptop-driven record/playback for the festicorn show.

Primary use: imported by the viewer server (audio-reactive/viewer/web/server.py).
Start the viewer, open the Recorder tab — the viewer lazily opens the
bridge serial via get_or_create_recorder() and serves /api/recorder/*
endpoints on its own origin.

The bridge ESP32 is optional: with no bridge the mic still records
to disk (WAV + .rmslog) and recordings still play back through the
laptop. The bridge is only needed to forward live frames or replay
LEDs to the receivers.

Standalone fallback (viewer down): `python festicorn/tools/mic_bridge.py --http`
exposes the same handlers on port 8765.

Pipeline:
  Mac mic ──→ RMS@25Hz ──→ SensorPacket ──┬─→ bridge (USB) → ESP-NOW → receiver
                  │                       └─→ .rmslog file
                  └──→ WAV file

Each .rmslog entry is a 25 Hz framed SensorPacket carrying mic RMS in
the rawRms tail (the IMU bytes are zero). During playback the laptop
replays packets to the bridge at original cadence and plays the WAV
in lockstep.

Frame format (both serial links): [0xA5][0x5A][LEN][PAYLOAD][XOR8]
SensorPacket payload (15 bytes, little-endian, packed):
  int16 ax,ay,az,gx,gy,gz; uint16 rawRms; uint8 micEnabled
"""

from __future__ import annotations
import argparse, glob, json, os, queue, struct, sys, threading, time, termios, tty, select, datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import serial
import sounddevice as sd
import soundfile as sf
import numpy as np

BAUD = 460800
SAMPLE_RATE = 16000
SENSOR_HZ = 25
RMS_SCALE_DEFAULT = 1_500_000.0  # float[-1,1] RMS → uint16; voice peaks ≈ 0.02 → ~30000
RECORDINGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))

LOG_EXT = ".rmslog"
LEGACY_LOG_EXTS = (".rmslog", ".ducklog")  # glob old recordings too
# Wire-format literal kept as b"DUCKLOG\x02" so existing recordings still load.
RECORDING_MAGIC = b"DUCKLOG\x02"

FRAME_M0, FRAME_M1 = 0xA5, 0x5A
PACKET_LEN = 15  # SensorPacket size
ZERO_IMU = b"\x00" * PACKET_LEN


def _strip_log_ext(path: str) -> str:
    for ext in LEGACY_LOG_EXTS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


# ── Frame protocol ─────────────────────────────────────────────────────────

class FrameParser:
    def __init__(self):
        self.s = 0; self.ln = 0; self.buf = bytearray()

    def feed(self, data: bytes):
        out = []
        for b in data:
            if self.s == 0:
                if b == FRAME_M0: self.s = 1
            elif self.s == 1:
                self.s = 2 if b == FRAME_M1 else 0
            elif self.s == 2:
                if b == 0 or b > 250: self.s = 0
                else: self.ln = b; self.buf = bytearray(); self.s = 3
            elif self.s == 3:
                self.buf.append(b)
                if len(self.buf) >= self.ln: self.s = 4
            elif self.s == 4:
                x = self.ln
                for byte in self.buf: x ^= byte
                if x == b: out.append(bytes(self.buf))
                self.s = 0
        return out


def encode_frame(payload: bytes) -> bytes:
    x = len(payload)
    for b in payload: x ^= b
    return bytes([FRAME_M0, FRAME_M1, len(payload)]) + payload + bytes([x])


def pack_packet(rms_u16: int, mic_enabled: int = 1) -> bytes:
    """Build a 15-byte SensorPacket with zero IMU and the given mic RMS."""
    return ZERO_IMU[:12] + struct.pack("<HB", min(max(rms_u16, 0), 65535), mic_enabled & 1)


# ── Port autodetect ────────────────────────────────────────────────────────

def identify_port(p: str, timeout_s: float = 1.5) -> str:
    """Return 'bridge' if serial emits the bridge banner (or is silent at baud),
    'unknown' otherwise."""
    try:
        with serial.Serial(p, BAUD, timeout=0.2) as s:
            time.sleep(0.2); s.reset_input_buffer()
            s.write(b'?')
            end = time.time() + timeout_s
            buf = b""
            while time.time() < end:
                if s.in_waiting: buf += s.read(s.in_waiting)
                time.sleep(0.05)
            text = bytes(c for c in buf if 32 <= c < 127 or c in (10,13)).decode('utf-8','replace')
            if "role=bridge" in text:
                return 'bridge'
            return 'bridge' if buf == b"" else 'unknown'
    except Exception:
        return 'unknown'


def discover_bridge_port(bridge_arg: str | None = None) -> str | None:
    """Return the bridge ESP32 serial port, or None if not found."""
    if bridge_arg:
        return bridge_arg
    cands = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*"))
    for p in cands:
        if identify_port(p) == 'bridge':
            return p
    return None


# ── Mic RMS source (rolling buffer) ────────────────────────────────────────

class MicRMS:
    def __init__(self, scale: float = RMS_SCALE_DEFAULT):
        self.scale = scale
        # ring of recent samples, ~80 ms worth
        self.window = np.zeros(SAMPLE_RATE * 80 // 1000, dtype=np.float32)
        self.idx = 0
        self.lock = threading.Lock()
        self.captured = 0  # total samples captured (for stats)
        # forwarded-to-WAV queue
        self.wav_q: queue.Queue[np.ndarray] = queue.Queue()
        self.recording = False

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
                    w[:n-first] = x[first:]
                self.idx = end % len(w)
            self.captured += n
        if self.recording:
            self.wav_q.put(indata.copy())

    def rms_u16(self) -> int:
        with self.lock:
            w = self.window
            r = float(np.sqrt(np.mean(w * w)))
        return int(min(65535, max(0, r * self.scale)))


# ── Recorder ───────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self, bridge_port: str | None, scale: float = RMS_SCALE_DEFAULT):
        self.bridge_port = bridge_port
        self.bridge = None
        self.mic = MicRMS(scale)
        self.audio_stream = None
        self.recording = False
        self.start_t = 0.0
        self.log_f = None
        self.log_path = None
        self.wav_path = None
        self.frame_count = 0
        self._stop = False
        self._tick_thread = None
        self._wav_thread = None
        self._wav_stop = threading.Event()
        # live status
        self.last_rms_u16 = 0
        self.last_packet_t = 0.0
        self.playing = False
        self._play_thread = None

    @property
    def has_bridge(self) -> bool:
        return self.bridge is not None

    def open(self):
        if self.bridge_port:
            try:
                self.bridge = serial.Serial(self.bridge_port, BAUD, timeout=0.05)
                time.sleep(0.3)
                self.bridge.reset_input_buffer()
            except Exception as e:
                print(f"[bridge open failed: {e}] — running mic-only")
                self.bridge = None
        # start mic stream immediately so RMS is hot when recording starts
        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32',
            callback=self.mic.callback, blocksize=512)
        self.audio_stream.start()
        self._stop = False
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def close(self):
        self._stop = True
        if self._tick_thread: self._tick_thread.join(timeout=1)
        if self.audio_stream:
            self.audio_stream.stop(); self.audio_stream.close(); self.audio_stream = None
        if self.bridge:
            self.bridge.close(); self.bridge = None

    # ── 25 Hz tick: mic RMS → SensorPacket → (bridge?) + (.rmslog?) ──────────
    def _tick_loop(self):
        next_tick = time.time()
        while not self._stop:
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now)); continue
            next_tick += 0.04
            rms = self.mic.rms_u16()
            self.last_rms_u16 = rms
            packet = pack_packet(rms, mic_enabled=1)
            if self.bridge:
                try:
                    self.bridge.write(encode_frame(packet))
                except Exception:
                    pass
            self.last_packet_t = time.time()
            if self.recording and self.log_f:
                t_us = int((time.time() - self.start_t) * 1_000_000)
                self.log_f.write(struct.pack("<QB", t_us, len(packet)) + packet)
                self.frame_count += 1

    # ── WAV writer ─────────────────────────────────────────────────────────
    def _wav_writer_loop(self, path: str):
        with sf.SoundFile(path, mode='w', samplerate=SAMPLE_RATE,
                          channels=1, subtype='PCM_16') as f:
            while not (self._wav_stop.is_set() and self.mic.wav_q.empty()):
                try:
                    chunk = self.mic.wav_q.get(timeout=0.1)
                    f.write(chunk[:, 0])
                except queue.Empty:
                    continue

    # ── Public ops ─────────────────────────────────────────────────────────
    def start(self) -> str:
        if self.recording: return "already recording"
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.join(RECORDINGS_DIR, ts)
        self.log_path = base + LOG_EXT
        self.wav_path = base + ".wav"
        self.log_f = open(self.log_path, "wb")
        self.log_f.write(RECORDING_MAGIC)
        self.frame_count = 0
        self.mic.captured = 0
        self.start_t = time.time()
        # drain wav queue
        while not self.mic.wav_q.empty():
            try: self.mic.wav_q.get_nowait()
            except queue.Empty: break
        self.mic.recording = True
        self._wav_stop = threading.Event()
        self._wav_thread = threading.Thread(target=self._wav_writer_loop,
                                            args=(self.wav_path,), daemon=True)
        self._wav_thread.start()
        self.recording = True
        return f"REC → {os.path.basename(base)}"

    def stop(self) -> str:
        if not self.recording: return "not recording"
        self.recording = False
        self.mic.recording = False
        self._wav_stop.set()
        if self._wav_thread: self._wav_thread.join(timeout=2)
        if self.log_f: self.log_f.close(); self.log_f = None
        dur = time.time() - self.start_t
        return (f"STOP {dur:.1f}s frames={self.frame_count} "
                f"audio_samples={self.mic.captured} → {os.path.basename(self.log_path)}")

    def latest_recording(self) -> str | None:
        logs = []
        for ext in LEGACY_LOG_EXTS:
            logs.extend(glob.glob(os.path.join(RECORDINGS_DIR, f"*{ext}")))
        return sorted(logs)[-1] if logs else None

    def _load_records(self, path: str):
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != RECORDING_MAGIC:
                return None, f"bad magic in {path} (expected {RECORDING_MAGIC!r}, got {magic!r})"
            records = []
            while True:
                hdr = f.read(9)
                if len(hdr) < 9: break
                t_us, ln = struct.unpack("<QB", hdr)
                payload = f.read(ln)
                if len(payload) < ln: break
                records.append((t_us, payload))
        return records, None

    def play_leds(self, log_path: str | None = None) -> str:
        """Replay LED serial frames only (no audio). Non-blocking — runs in a thread."""
        if not self.bridge: return "no bridge attached"
        if self.recording: return "stop recording first"
        if self.playing: return "already playing"
        path = log_path or self.latest_recording()
        if not path or not os.path.exists(path):
            return "no recording found"
        records, err = self._load_records(path)
        if err: return err
        if not records: return "empty recording"

        def _replay():
            self.playing = True
            try:
                t0 = time.time()
                for t_us, payload in records:
                    if not self.playing: break
                    target = t0 + t_us / 1_000_000
                    now = time.time()
                    if target > now: time.sleep(target - now)
                    try:
                        self.bridge.write(encode_frame(payload))
                    except Exception:
                        pass
            finally:
                self.playing = False

        self._play_thread = threading.Thread(target=_replay, daemon=True)
        self._play_thread.start()
        dur_s = records[-1][0] / 1_000_000 if records else 0
        return f"PLAY_LEDS {len(records)} frames, {dur_s:.1f}s"

    def stop_playback(self):
        self.playing = False
        if self._play_thread:
            self._play_thread.join(timeout=2)
            self._play_thread = None

    def playback(self, log_path: str | None = None) -> str:
        if self.recording: return "stop recording first"
        path = log_path or self.latest_recording()
        if not path or not os.path.exists(path):
            return "no recording found"
        wav = _strip_log_ext(path) + ".wav"
        records, err = self._load_records(path)
        if err: return err
        if not records: return "empty recording"

        # Open a dedicated OutputStream for playback. Coexists with the
        # open InputStream (full duplex) and avoids sd.play's process-wide
        # global state, which previously SIGTRAPped on the second call.
        wav_data = None
        if os.path.exists(wav):
            try:
                wav_data, wav_sr = sf.read(wav, dtype='float32')
                if wav_data.ndim == 1:
                    wav_data = wav_data.reshape(-1, 1)
            except Exception as e:
                print(f"\n[wav read error: {e}]")
                wav_data = None

        out_stream = None
        audio_thread = None
        if wav_data is not None:
            try:
                out_stream = sd.OutputStream(samplerate=wav_sr,
                                             channels=wav_data.shape[1],
                                             dtype='float32',
                                             blocksize=1024)
                out_stream.start()
            except Exception as e:
                print(f"\n[output stream open failed: {e}]")
                out_stream = None
        if out_stream is not None:
            def _play():
                try:
                    out_stream.write(wav_data)
                except Exception as e:
                    print(f"\n[audio write error: {e}]")
            audio_thread = threading.Thread(target=_play, daemon=True)
            audio_thread.start()

        try:
            t0 = time.time()
            for t_us, payload in records:
                target = t0 + t_us / 1_000_000
                now = time.time()
                if target > now: time.sleep(target - now)
                if self.bridge:
                    try:
                        self.bridge.write(encode_frame(payload))
                    except Exception:
                        pass
            if audio_thread: audio_thread.join()
            dur = time.time() - t0
        finally:
            if out_stream is not None:
                try: out_stream.stop(); out_stream.close()
                except Exception: pass
        return f"PLAY {dur:.1f}s frames={len(records)} ({os.path.basename(path)})"


# ── Single-key terminal UI ─────────────────────────────────────────────────

def getkey_nb() -> str | None:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


HTTP_PORT = 8765

class RecorderHandler(BaseHTTPRequestHandler):
    recorder: Recorder  # set on class before serving

    def log_message(self, fmt, *a):
        pass  # silence per-request logs

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        rec = self.recorder

        if path == "/api/status":
            dur = (time.time() - rec.start_t) if rec.recording else 0.0
            has_rec = rec.latest_recording() is not None
            rec_dur = None
            if has_rec:
                lp = rec.latest_recording()
                wav = _strip_log_ext(lp) + ".wav"
                if os.path.exists(wav):
                    try:
                        info = sf.info(wav)
                        rec_dur = info.duration
                    except Exception:
                        pass
            self._json(200, {
                "state": "recording" if rec.recording else ("playing" if rec.playing else "idle"),
                "timer": round(dur, 1),
                "rms": rec.last_rms_u16,
                "has_recording": has_rec,
                "recording_duration": rec_dur,
                "frame_count": rec.frame_count,
                "has_bridge": rec.has_bridge,
            })
            return

        if path == "/api/recording/wav":
            lp = rec.latest_recording()
            if not lp:
                self._json(404, {"error": "no recording"})
                return
            wav = _strip_log_ext(lp) + ".wav"
            if not os.path.exists(wav):
                self._json(404, {"error": "wav not found"})
                return
            with open(wav, "rb") as f:
                data = f.read()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        rec = self.recorder

        if path == "/api/record":
            self._json(200, {"msg": rec.start()})
            return
        if path == "/api/stop":
            self._json(200, {"msg": rec.stop()})
            return
        if path == "/api/play":
            if not rec.has_bridge:
                self._json(503, {"error": "no bridge attached"})
                return
            self._json(200, {"msg": rec.play_leds()})
            return
        if path == "/api/play/stop":
            rec.stop_playback()
            self._json(200, {"msg": "stopped"})
            return

        self._json(404, {"error": "not found"})

    def do_DELETE(self):
        path = urlparse(self.path).path
        rec = self.recorder

        if path == "/api/recording":
            lp = rec.latest_recording()
            if not lp:
                self._json(404, {"error": "no recording"})
                return
            wav = _strip_log_ext(lp) + ".wav"
            for p in (lp, wav):
                try: os.remove(p)
                except OSError: pass
            self._json(200, {"msg": "deleted"})
            return

        self._json(404, {"error": "not found"})


# ── Lazy singleton for server.py import path ───────────────────────────────

_singleton: Recorder | None = None
_singleton_lock = threading.Lock()


def get_or_create_recorder() -> Recorder:
    """Return a process-wide Recorder, opening mic + (optional) bridge serial
    on first call. Mic failures raise; missing bridge just disables forwarding.
    """
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        bridge_port = discover_bridge_port()
        rec = Recorder(bridge_port, RMS_SCALE_DEFAULT)
        rec.open()
        _singleton = rec
        return _singleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-port", help="espnow-bridge serial port (omit for mic-only)")
    ap.add_argument("--play", help=f"playback this {LOG_EXT} and exit")
    ap.add_argument("--http", type=int, nargs="?", const=HTTP_PORT, default=None,
                    metavar="PORT", help=f"start HTTP API (default port {HTTP_PORT})")
    args = ap.parse_args()

    bridge_port = discover_bridge_port(args.bridge_port)
    print(f"bridge : {bridge_port or '(none — mic-only)'}")
    rec = Recorder(bridge_port, RMS_SCALE_DEFAULT)
    rec.open()

    if args.http is not None:
        RecorderHandler.recorder = rec
        httpd = HTTPServer(("0.0.0.0", args.http), RecorderHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"http   : http://localhost:{args.http}")
    if args.play:
        print(rec.playback(args.play)); rec.close(); return

    print("commands:  r=record  s=stop  p=playback last  q=quit")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        last_t = 0
        while True:
            k = getkey_nb()
            if k:
                if k == 'r':
                    msg = rec.start()
                elif k == 's':
                    msg = rec.stop()
                elif k == 'p':
                    print("\rplaying…" + " " * 60, flush=True)
                    msg = rec.playback()
                elif k == 'q':
                    if rec.recording: rec.stop()
                    break
                else:
                    msg = None
                if msg: print("\r" + " " * 80 + "\r" + msg, flush=True)
            if time.time() - last_t > 0.2:
                last_t = time.time()
                state = "REC" if rec.recording else "live"
                dur = (time.time() - rec.start_t) if rec.recording else 0.0
                sys.stdout.write(f"\r{state}  rms={rec.last_rms_u16:5d}  "
                                 f"frames={rec.frame_count:4d}  t={dur:5.1f}s    ")
                sys.stdout.flush()
            time.sleep(0.02)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        rec.close()
        print("\nbye")


if __name__ == "__main__":
    main()
