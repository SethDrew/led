#!/usr/bin/env python3
"""
mic_echo.py — real-time mic echo-back for INMP441 profiling.

Reads framed binary blocks (AA 55 A5 5A | u16 block# | u16 n | n*int16le)
from the ESP32-C3 mic_profile firmware, optionally processes the audio
(passthrough / WebRTC VAD + spectral gate / stationary spectral gate),
and plays it through the default output device (headphones).

NOTE: latency is ~0.5-1s due to the chunked processing buffer needed
for noisereduce/VAD context. That's expected; this is a profiling tool,
not a monitoring path.

Usage (run inside the project venv):
  .venv/bin/python tools/mic_echo.py --port /dev/cu.usbmodem21401 --algo raw
  .venv/bin/python tools/mic_echo.py --port /dev/cu.usbmodem21401 --algo webrtc --agg 2 --strength 0.8
  .venv/bin/python tools/mic_echo.py --port /dev/cu.usbmodem21401 --algo denoise --strength 0.7

Algos:
  raw     — passthrough, no processing
  denoise — spectral noise reduction only (no voice gate). Learns noise
            profile from first 2s, then subtracts. Passes everything
            (voice, music, hums, claps) but cleaner.
  webrtc_ns — REAL WebRTC noise suppression (Wiener filter). Uses
            --agg as NS level: 0=mild, 1=moderate, 2=high, 3=very high.
            This is what phones actually use. No voice gate.
  rnnoise — Mozilla RNNoise neural denoiser. No tunable params.
  webrtc  — noisereduce spectral gating + VAD voice gate (legacy).
"""
import argparse
import math
import queue
import signal
import struct
import sys
import threading
import time

import numpy as np
import serial
import sounddevice as sd

SYNC = b"\xAA\x55\xA5\x5A"
SAMPLE_RATE = 16000


def find_sync(ser, leftover=b""):
    buf = leftover
    while True:
        idx = buf.find(SYNC)
        if idx >= 0:
            preamble = buf[:idx]
            if preamble:
                for line in preamble.split(b"\n"):
                    s = line.strip()
                    if s.startswith(b"#"):
                        sys.stderr.write(s.decode("utf-8", "replace") + "\n")
                        sys.stderr.flush()
            return buf[idx + 4:]
        if len(buf) > 4:
            *lines, tail = buf.split(b"\n")
            for line in lines:
                s = line.strip()
                if s.startswith(b"#"):
                    sys.stderr.write(s.decode("utf-8", "replace") + "\n")
            buf = tail
        chunk = ser.read(256)
        if not chunk:
            continue
        buf += chunk


def apply_gain(samples_f32, gain_db):
    if gain_db == 0.0:
        return samples_f32
    g = 10.0 ** (gain_db / 20.0)
    return samples_f32 * g


def process_raw(chunk_f32, state, args):
    return chunk_f32


def process_denoise(chunk_f32, state, args):
    import noisereduce as nr
    if state.get("noise_profile") is None:
        state.setdefault("noise_accum", []).append(chunk_f32.copy())
        accumulated = sum(len(c) for c in state["noise_accum"])
        if accumulated >= SAMPLE_RATE * 2:
            state["noise_profile"] = np.concatenate(state["noise_accum"])
            sys.stderr.write("\n# noise profile captured (denoise)\n")
    if state.get("noise_profile") is not None:
        return nr.reduce_noise(
            y=chunk_f32,
            sr=SAMPLE_RATE,
            y_noise=state["noise_profile"],
            stationary=False,
            prop_decrease=args.strength,
        ).astype(np.float32)
    return nr.reduce_noise(
        y=chunk_f32,
        sr=SAMPLE_RATE,
        stationary=True,
        prop_decrease=args.strength,
    ).astype(np.float32)


def process_webrtc(chunk_f32, state, args):
    import noisereduce as nr
    import webrtcvad

    vad = state.get("vad")
    if vad is None:
        vad = webrtcvad.Vad(args.agg)
        state["vad"] = vad

    # Build/refresh noise profile from first ~2s of audio.
    if state.get("noise_profile") is None:
        state.setdefault("noise_accum", []).append(chunk_f32.copy())
        accumulated = sum(len(c) for c in state["noise_accum"])
        if accumulated >= SAMPLE_RATE * 2:
            state["noise_profile"] = np.concatenate(state["noise_accum"])
            sys.stderr.write("\n# noise profile captured\n")

    # Spectral gate against stored noise profile (or stationary if not ready).
    if state.get("noise_profile") is not None:
        denoised = nr.reduce_noise(
            y=chunk_f32,
            sr=SAMPLE_RATE,
            y_noise=state["noise_profile"],
            stationary=False,
            prop_decrease=args.strength,
        ).astype(np.float32)
    else:
        denoised = nr.reduce_noise(
            y=chunk_f32,
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=args.strength,
        ).astype(np.float32)

    # VAD gate: 20ms frames @ 16kHz = 320 samples. Mute frames marked non-speech.
    frame_len = 320
    pcm16 = np.clip(denoised * 32768.0, -32768, 32767).astype(np.int16)
    out = np.zeros_like(denoised)
    n_frames = len(denoised) // frame_len
    for i in range(n_frames):
        s = i * frame_len
        e = s + frame_len
        frame_bytes = pcm16[s:e].tobytes()
        try:
            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception:
            is_speech = True
        if is_speech:
            out[s:e] = denoised[s:e]
    # tail: pass through whatever frame didn't fit
    tail_start = n_frames * frame_len
    if tail_start < len(denoised):
        out[tail_start:] = denoised[tail_start:]
    return out


def process_webrtc_ns(chunk_f32, state, args):
    from webrtc_noise_gain import AudioProcessor
    ap = state.get("ap")
    if ap is None:
        ns_level = args.agg  # reuse agg param as NS level 0-3
        ap = AudioProcessor(SAMPLE_RATE, ns_level)
        state["ap"] = ap
    # Process in 10ms frames (160 samples at 16kHz)
    frame_len = 160
    pcm16 = np.clip(chunk_f32 * 32768.0, -32768, 32767).astype(np.int16)
    out_frames = []
    for i in range(0, len(pcm16) - frame_len + 1, frame_len):
        frame = pcm16[i:i + frame_len]
        result = ap.Process10ms(frame.tobytes())
        out_frames.append(np.frombuffer(result.audio, dtype=np.int16))
    if not out_frames:
        return chunk_f32
    return np.concatenate(out_frames).astype(np.float32) / 32768.0


def process_rnnoise(chunk_f32, state, args):
    from pyrnnoise import RNNoise
    rn = state.get("rn")
    if rn is None:
        rn = RNNoise(SAMPLE_RATE)
        state["rn"] = rn
    # pyrnnoise expects int16-scale input, outputs int16-scale
    chunk_i16 = chunk_f32 * 32768.0
    out_frames = []
    for vad_prob, frame in rn.denoise_chunk(chunk_i16):
        out_frames.append(frame.flatten())
    if not out_frames:
        return chunk_f32
    result = np.concatenate(out_frames).astype(np.float32) / 32768.0
    return result


def _autocorr_peak(frame):
    # Returns (peak_score 0..1, lag) for periodicity in 60-500 Hz range.
    n = len(frame)
    if n < 64:
        return 0.0, 0
    f = frame - np.mean(frame)
    energy = float(np.dot(f, f))
    if energy < 1e-8:
        return 0.0, 0
    # lag range for 60-500 Hz at 16 kHz
    min_lag = max(2, SAMPLE_RATE // 500)
    max_lag = min(n - 1, SAMPLE_RATE // 60)
    if max_lag <= min_lag:
        return 0.0, 0
    ac = np.correlate(f, f, mode="full")
    mid = len(ac) // 2
    region = ac[mid + min_lag:mid + max_lag + 1]
    if len(region) == 0:
        return 0.0, 0
    peak = float(np.max(region))
    lag = int(np.argmax(region)) + min_lag
    score = max(0.0, min(1.0, peak / energy))
    return score, lag


def process_intent(chunk_f32, state, args):
    from webrtc_noise_gain import AudioProcessor

    # Stage 1: WebRTC NS (agg=3) for clean input
    ap = state.get("ap")
    if ap is None:
        ap = AudioProcessor(SAMPLE_RATE, 3)
        state["ap"] = ap
    frame_len_ns = 160  # 10ms
    pcm16 = np.clip(chunk_f32 * 32768.0, -32768, 32767).astype(np.int16)
    out_frames = []
    for i in range(0, len(pcm16) - frame_len_ns + 1, frame_len_ns):
        frame = pcm16[i:i + frame_len_ns]
        result = ap.Process10ms(frame.tobytes())
        out_frames.append(np.frombuffer(result.audio, dtype=np.int16))
    if out_frames:
        clean = np.concatenate(out_frames).astype(np.float32) / 32768.0
    else:
        clean = chunk_f32

    # Weights
    weights = state.get("weights")
    if weights is None:
        if getattr(args, "intent_weights", None):
            try:
                parts = [float(x) for x in args.intent_weights.split(",")]
                if len(parts) == 4:
                    weights = parts
            except Exception:
                weights = None
        if weights is None:
            weights = [0.35, 0.20, 0.25, 0.20]
        state["weights"] = weights

    # 20ms analysis frames @ 16kHz = 320 samples
    frame_len = 320
    n_frames = max(1, len(clean) // frame_len)

    # Persistent state
    prev_rms_db = state.get("prev_rms_db", -120.0)
    sustain_samples = state.get("sustain_samples", 0)
    in_event = state.get("in_event", False)
    onset_log = state.setdefault("onset_log", [])  # list of (time, energy)
    sample_clock = state.get("sample_clock", 0)
    last_print = state.get("last_print", 0.0)
    noise_floor_db = state.get("noise_floor_db", -60.0)

    harm_acc = 0.0
    atk_acc = 0.0
    sus_acc = 0.0
    rep_acc = 0.0
    lvl_acc = 0.0
    confidences = np.zeros(n_frames, dtype=np.float32)
    rms_dbs = np.zeros(n_frames, dtype=np.float32)

    NOISE_FLOOR_ALPHA = 0.001
    EVENT_HYSTERESIS_DB = 6.0

    # Energy envelope — tracks RMS of active frames (in dB)
    # Slow EMA learns "typical intentional speaker level"
    # Frames far below get attenuated proportionally
    frame_dur = frame_len / SAMPLE_RATE
    energy_env_db = state.get("energy_env_db", -40.0)
    energy_env_frames = state.get("energy_env_frames", 0)
    alpha_energy_rise = 1.0 - math.exp(-frame_dur / 0.500)   # 500ms to track up
    alpha_energy_fall = 1.0 - math.exp(-frame_dur / 3.000)   # 3s to forget down
    LEVEL_RANGE_DB = 20.0  # frames this far below envelope → level_score=0

    for fi in range(n_frames):
        s = fi * frame_len
        e = s + frame_len
        frame = clean[s:e]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        rms_db = 20.0 * math.log10(max(rms, 1e-6))
        rms_dbs[fi] = rms_db

        if rms_db < noise_floor_db + EVENT_HYSTERESIS_DB:
            noise_floor_db = (1.0 - NOISE_FLOOR_ALPHA) * noise_floor_db + NOISE_FLOOR_ALPHA * rms_db

        above_floor = rms_db > (noise_floor_db + EVENT_HYSTERESIS_DB)

        if not above_floor:
            if in_event:
                in_event = False
                sustain_samples = 0
            prev_rms_db = rms_db
            confidences[fi] = 0.0
            continue

        # Update energy envelope (asymmetric, only on active frames)
        ae = alpha_energy_rise if rms_db > energy_env_db else alpha_energy_fall
        energy_env_db += ae * (rms_db - energy_env_db)
        energy_env_frames += 1

        # 1. Harmonic
        harm, _lag = _autocorr_peak(frame)
        harm_score = harm

        # 2. Attack
        delta_db = rms_db - prev_rms_db
        if delta_db < 30.0:
            attack_score = 1.0
        else:
            attack_score = max(0.0, 1.0 - (delta_db - 30.0) / 20.0)
        prev_rms_db = rms_db

        # Event tracking
        frame_time_now = (sample_clock + s) / float(SAMPLE_RATE)
        if not in_event:
            in_event = True
            sustain_samples = 0
            onset_log.append((frame_time_now, rms))
            cutoff = frame_time_now - 5.0
            while onset_log and onset_log[0][0] < cutoff:
                onset_log.pop(0)
        sustain_samples += frame_len

        # 3. Sustain
        sustain_ms = (sustain_samples / SAMPLE_RATE) * 1000.0
        sustain_score = max(0.0, min(1.0, sustain_ms / 150.0))

        # 4. Repetition
        n_recent = len(onset_log)
        if n_recent >= 3:
            repetition_score = 1.0
        elif n_recent == 2:
            repetition_score = 0.6
        else:
            repetition_score = 0.2

        # 5. Level — proximity to learned energy envelope
        if energy_env_frames < 25:
            level_score = 1.0  # permissive during cold start
        else:
            gap_db = energy_env_db - rms_db  # positive = below envelope
            if gap_db <= 0.0:
                level_score = 1.0  # at or above envelope
            else:
                level_score = max(0.0, 1.0 - gap_db / LEVEL_RANGE_DB)

        # Multiplicative: level gates the whole thing
        core = harm_score * sustain_score * attack_score * level_score
        boosted = core + 0.3 * repetition_score * (1.0 - core)
        confidence = boosted * boosted
        confidence = max(0.0, min(1.0, confidence))

        confidences[fi] = confidence
        harm_acc += harm_score
        atk_acc += attack_score
        sus_acc += sustain_score
        rep_acc += repetition_score
        lvl_acc += level_score

    # Save state
    state["prev_rms_db"] = prev_rms_db
    state["sustain_samples"] = sustain_samples
    state["in_event"] = in_event
    state["sample_clock"] = sample_clock + len(clean)
    state["noise_floor_db"] = noise_floor_db
    state["energy_env_db"] = energy_env_db
    state["energy_env_frames"] = energy_env_frames

    # Apply confidence as gain envelope
    gain = np.ones_like(clean)
    for fi in range(n_frames):
        s = fi * frame_len
        e = min(s + frame_len, len(clean))
        gain[s:e] = confidences[fi]
    # One-pole sample-level smoothing
    if n_frames > 1:
        smoothed = np.empty_like(gain)
        prev_g = state.get("prev_gain", confidences[0] if n_frames else 0.0)
        alpha = 0.02
        for i in range(len(gain)):
            prev_g = prev_g + alpha * (gain[i] - prev_g)
            smoothed[i] = prev_g
        state["prev_gain"] = prev_g
        gain = smoothed

    out = clean * gain

    # Print breakdown every 250ms
    now = time.time()
    if now - last_print >= 0.25 and n_frames > 0:
        avg_conf = float(np.mean(confidences))
        active = n_frames - np.count_nonzero(confidences == 0.0)
        sys.stderr.write(
            f"\nconf={avg_conf:.2f} "
            f"[harm={harm_acc / max(active,1):.2f} "
            f"atk={atk_acc / max(active,1):.2f} sus={sus_acc / max(active,1):.2f} "
            f"rep={rep_acc / max(active,1):.2f} lvl={lvl_acc / max(active,1):.2f}] "
            f"energy_env={energy_env_db:.1f}dB floor={noise_floor_db:.1f}dB\n"
        )
        sys.stderr.flush()
        state["last_print"] = now

    return out


ALGOS = {
    "raw": process_raw,
    "denoise": process_denoise,
    "webrtc": process_webrtc,
    "webrtc_ns": process_webrtc_ns,
    "rnnoise": process_rnnoise,
    "intent": process_intent,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--port", default="/dev/cu.usbmodem21401")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--algo", choices=list(ALGOS.keys()), default="raw")
    ap.add_argument("--agg", type=int, default=2,
                    help="WebRTC VAD aggressiveness 0-3 (default 2)")
    ap.add_argument("--strength", type=float, default=None,
                    help="Noise reduction prop_decrease 0.0-1.0 "
                         "(default 0.8 for webrtc, 0.7 for denoise)")
    ap.add_argument("--gain", type=float, default=0.0,
                    help="Output gain in dB (default 0)")
    ap.add_argument("--intent-weights", type=str, default=None,
                    help="Intent algo weights: 4 comma-separated floats "
                         "(harmonic,attack,sustain,repetition). "
                         "Default: 0.35,0.20,0.25,0.20")
    ap.add_argument("--chunk-sec", type=float, default=0.5,
                    help="Processing chunk size in seconds (default 0.5). "
                         "Larger = more context for denoise, more latency.")
    args = ap.parse_args()

    if args.strength is None:
        args.strength = 0.8 if args.algo == "webrtc" else 0.7
    args.strength = max(0.0, min(1.0, args.strength))
    args.agg = max(0, min(3, args.agg))

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        sys.stderr.write(f"# ERROR: cannot open {args.port}: {e}\n")
        sys.exit(1)
    sys.stderr.write(
        f"# Opened {args.port} @ {args.baud}, algo={args.algo} "
        f"strength={args.strength} agg={args.agg} gain={args.gain}dB "
        f"chunk={args.chunk_sec}s\n"
    )
    sys.stderr.write(
        "# NOTE: ~0.5-1s latency from processing buffer is expected.\n"
    )

    chunk_n = int(SAMPLE_RATE * args.chunk_sec)
    play_q = queue.Queue(maxsize=8)
    stopping = {"flag": False}

    def stop(*_):
        stopping["flag"] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    # ---------- audio output thread ----------
    def audio_callback(outdata, frames, time_info, status):
        if status:
            sys.stderr.write(f"\n# sd status: {status}\n")
        try:
            block = play_q.get_nowait()
        except queue.Empty:
            outdata[:] = 0
            return
        if len(block) < frames:
            outdata[:len(block), 0] = block
            outdata[len(block):, 0] = 0
        else:
            outdata[:, 0] = block[:frames]

    out_block = max(256, chunk_n)

    # ---------- processing thread ----------
    process_fn = ALGOS[args.algo]
    proc_state = {}
    raw_q = queue.Queue(maxsize=64)

    def processor():
        accum = np.zeros(0, dtype=np.float32)
        while not stopping["flag"]:
            try:
                block = raw_q.get(timeout=0.2)
            except queue.Empty:
                continue
            accum = np.concatenate([accum, block])
            while len(accum) >= chunk_n:
                chunk = accum[:chunk_n]
                accum = accum[chunk_n:]
                try:
                    processed = process_fn(chunk, proc_state, args)
                except Exception as e:
                    sys.stderr.write(f"\n# process error: {e}\n")
                    processed = chunk
                processed = apply_gain(processed, args.gain)
                processed = np.clip(processed, -1.0, 1.0).astype(np.float32)
                try:
                    play_q.put(processed, timeout=1.0)
                except queue.Full:
                    pass

    proc_thread = threading.Thread(target=processor, daemon=True)
    proc_thread.start()

    # ---------- serial reader (main thread) ----------
    leftover = b""
    last_block = None
    drops = 0
    last_meter_t = time.time()
    sumsq_window = 0.0
    n_window = 0

    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=out_block,
            callback=audio_callback,
        ):
            while not stopping["flag"]:
                leftover = find_sync(ser, leftover)
                while len(leftover) < 4:
                    chunk = ser.read(4 - len(leftover))
                    if not chunk:
                        if stopping["flag"]:
                            break
                        continue
                    leftover += chunk
                if len(leftover) < 4:
                    break
                block_no, n_samples = struct.unpack("<HH", leftover[:4])
                leftover = leftover[4:]

                if n_samples == 0 or n_samples > 4096:
                    drops += 1
                    continue

                need = n_samples * 2
                while len(leftover) < need:
                    chunk = ser.read(need - len(leftover))
                    if not chunk:
                        if stopping["flag"]:
                            break
                        continue
                    leftover += chunk
                if len(leftover) < need:
                    break
                payload = leftover[:need]
                leftover = leftover[need:]

                if last_block is not None:
                    expected = (last_block + 1) & 0xFFFF
                    if block_no != expected:
                        gap = (block_no - expected) & 0xFFFF
                        drops += gap
                last_block = block_no

                pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                try:
                    raw_q.put(pcm, timeout=0.5)
                except queue.Full:
                    pass

                sumsq_window += float(np.sum(pcm.astype(np.float64) ** 2)) * (32768.0 ** 2)
                n_window += len(pcm)

                now = time.time()
                if now - last_meter_t >= 0.25:
                    rms = math.sqrt(sumsq_window / n_window) if n_window else 0.0
                    dbfs = 20 * math.log10(rms / 32768.0) if rms > 0 else -120.0
                    bars = int(min(40, max(0, (dbfs + 60) * 40 / 60)))
                    meter = "#" * bars + "-" * (40 - bars)
                    sys.stderr.write(
                        f"\rrms={rms:7.0f}  {dbfs:6.1f} dBFS  [{meter}]  "
                        f"drops={drops}  qraw={raw_q.qsize()} qplay={play_q.qsize()}"
                    )
                    sys.stderr.flush()
                    sumsq_window = 0.0
                    n_window = 0
                    last_meter_t = now
    finally:
        stopping["flag"] = True
        ser.close()
        sys.stderr.write(f"\n# stopped. drops={drops}\n")


if __name__ == "__main__":
    main()
