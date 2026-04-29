"""
Cut + align dual-capture audio session into per-segment dataset.

Reads inmp441.wav (mic) and system.wav (BlackHole loopback), detects songs by
silence boundaries on system.wav, matches them to narration markers, aligns the
mic to system via cross-correlation, gain-matches the mic, and writes per-
segment WAV pairs + metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

SESSION_DIR = Path(
    "/Users/sethdrew/Documents/projects/led/library/test-vectors/"
    "inmp441-validation/session_20260428_171620"
)
SR = 16000

# Silence detection (on system.wav)
SIL_HOP_S = 0.10           # 100 ms RMS hop
SIL_THRESH_DBFS = -45.0    # below this = silent
SIL_MIN_GAP_S = 1.0        # silent run length to count as inter-region gap
MIN_SONG_S = 5.0           # discard sub-5s active spans as transient

# Envelope cross-correlation
ENV_HOP_S = 0.010          # 10 ms RMS hop -> ~100 Hz envelope
ENV_HZ = 1.0 / ENV_HOP_S
LOCAL_LAG_SEARCH_S = 2.0   # ±2 s refinement window around global lag prior

# Mic gain matching
PEAK_PCTILE = 99.5

# Voice segment (mic-only) detection
VOICE_PRE_S = 5.0          # search window before narration t_offset
VOICE_POST_S = 60.0        # search window after narration t_offset
VOICE_RMS_THRESH_DBFS = -45.0
VOICE_MIN_GAP_S = 1.0


@dataclass
class Song:
    start_s: float          # in system.wav timeline (== t_offset since BlackHole started ~start_iso)
    end_s: float
    @property
    def dur_s(self) -> float:
        return self.end_s - self.start_s
    @property
    def mid_s(self) -> float:
        return 0.5 * (self.start_s + self.end_s)


def rms_envelope(x: np.ndarray, sr: int, hop_s: float) -> np.ndarray:
    """Block-wise RMS using non-overlapping windows of length hop_s."""
    win = max(1, int(round(hop_s * sr)))
    n_blocks = len(x) // win
    if n_blocks == 0:
        return np.zeros(0, dtype=np.float32)
    trimmed = x[: n_blocks * win].reshape(n_blocks, win).astype(np.float32)
    return np.sqrt(np.mean(trimmed * trimmed, axis=1) + 1e-20)


def db(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(x, 1e-12))


def find_song_in_window(system: np.ndarray, sr: int,
                        win_start_s: float, win_end_s: float,
                        marker_t_s: float | None = None
                        ) -> Song | None:
    """Within [win_start_s, win_end_s] of system.wav, find the contiguous
    musically-active region that starts soonest after `marker_t_s` (markers
    fire just before playback). If `marker_t_s` is None, picks the longest."""
    a = max(0, int(round(win_start_s * sr)))
    b = min(len(system), int(round(win_end_s * sr)))
    if b - a <= 0:
        return None
    chunk = system[a:b]
    env = rms_envelope(chunk, sr, SIL_HOP_S)
    env_db_arr = db(env)
    active = env_db_arr > SIL_THRESH_DBFS
    n = len(active)
    if n == 0:
        return None
    min_gap_frames = int(round(SIL_MIN_GAP_S / SIL_HOP_S))

    # find silent runs >= min_gap_frames
    silent_runs: list[tuple[int, int]] = []
    j = 0
    while j < n:
        if not active[j]:
            k = j
            while k < n and not active[k]:
                k += 1
            if (k - j) >= min_gap_frames:
                silent_runs.append((j, k))
            j = k
        else:
            j += 1

    # active regions = the inter-silence stretches (and trim leading/trailing silence)
    regions: list[tuple[int, int]] = []
    prev = 0
    for (s, e) in silent_runs:
        if s > prev:
            regions.append((prev, s))
        prev = e
    if prev < n:
        regions.append((prev, n))

    # filter: must have audible peak above threshold + 6dB and length >= MIN_SONG_S
    candidates: list[tuple[int, int]] = []
    for (rs, re) in regions:
        if (re - rs) * SIL_HOP_S < MIN_SONG_S:
            continue
        if np.max(env_db_arr[rs:re]) <= SIL_THRESH_DBFS + 6.0:
            continue
        # also trim leading/trailing low-energy frames inside the region
        active_local = active[rs:re]
        # tighten to first/last active sample
        sub_act = np.where(active_local)[0]
        if len(sub_act) == 0:
            continue
        candidates.append((rs + int(sub_act[0]), rs + int(sub_act[-1]) + 1))

    if not candidates:
        return None
    if marker_t_s is None:
        rs, re = max(candidates, key=lambda r: r[1] - r[0])
    else:
        # Take all candidates at-or-after the marker and merge them into one
        # span (since they all fall in a single labeled song's window).
        marker_offset_frames = int(round((marker_t_s - win_start_s) / SIL_HOP_S))
        after = [c for c in candidates if c[0] >= marker_offset_frames - 5]
        if after:
            rs = min(c[0] for c in after)
            re = max(c[1] for c in after)
        else:
            rs, re = max(candidates, key=lambda r: r[1] - r[0])
    start_s = win_start_s + rs * SIL_HOP_S
    end_s = win_start_s + re * SIL_HOP_S
    return Song(start_s=start_s, end_s=end_s)


def xcorr_lag_envelope(env_a: np.ndarray, env_b: np.ndarray,
                       max_lag_frames: int | None = None) -> tuple[int, float]:
    """Cross-correlate two envelopes via FFT.

    Returns (lag_frames, peak) where positive `lag_frames` means env_b lags
    env_a; i.e. shifting env_b LEFT by `lag_frames` aligns it with env_a.
    Equivalently: env_a[t] ≈ env_b[t + lag].
    """
    a = env_a.astype(np.float64) - np.mean(env_a)
    b = env_b.astype(np.float64) - np.mean(env_b)
    la, lb = len(a), len(b)
    n = la + lb - 1
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    B = np.fft.rfft(b, n=nfft)
    c_full = np.fft.irfft(A * np.conj(B), n=nfft)
    # We want c[k] = sum_t a[t] * b[t + k] for k in [-(lb-1) .. (la-1)].
    # FFT convention with conj(B): c_full[k] for k>=0 is the lag-k value;
    # c_full[nfft - k] for k>0 is the lag -k value.
    # So valid lags live at indices [0..la-1] (positive) and [nfft-(lb-1)..nfft-1] (negative).
    pos = c_full[: la]                       # lags 0 .. la-1
    neg = c_full[nfft - (lb - 1):] if lb > 1 else np.zeros(0)  # lags -(lb-1) .. -1
    c_lin = np.concatenate([neg, pos])       # lag axis from -(lb-1) to la-1
    lags = np.arange(-(lb - 1), la)
    assert len(c_lin) == len(lags)
    if max_lag_frames is not None:
        mask = np.abs(lags) <= max_lag_frames
        c_use = c_lin[mask]
        lags_use = lags[mask]
    else:
        c_use = c_lin
        lags_use = lags
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    c_use = c_use / norm
    idx = int(np.argmax(c_use))
    return int(lags_use[idx]), float(c_use[idx])


def global_stream_lag_seconds(system: np.ndarray, mic: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate how many seconds the mic stream lags behind the system stream.

    Convention: returns L such that  sys[t] ≈ mic[t - L]  (mic content at time τ
    matches sys content at time τ + L). Equivalently: mic sample 0 corresponds
    to sys sample at L seconds into sys.wav.

    Method: full broadband log-envelope FFT xcorr at 20 Hz. The strongest peak
    is the global stream lag.
    """
    hop = 0.05
    win = max(1, int(round(hop * sr)))
    n_sys = len(system) // win
    n_mic = len(mic) // win
    sys_e = np.sqrt(np.mean(system[: n_sys * win].reshape(n_sys, win)
                            .astype(np.float64) ** 2, axis=1) + 1e-20)
    mic_e = np.sqrt(np.mean(mic[: n_mic * win].reshape(n_mic, win)
                            .astype(np.float64) ** 2, axis=1) + 1e-20)
    a = np.log(sys_e + 1e-6); a -= a.mean()
    b = np.log(mic_e + 1e-6); b -= b.mean()
    la, lb = len(a), len(b)
    nfft = 1 << (la + lb - 1 - 1).bit_length()
    A = np.fft.rfft(a, n=nfft); B = np.fft.rfft(b, n=nfft)
    c_full = np.fft.irfft(A * np.conj(B), n=nfft)
    pos = c_full[:la]
    neg = c_full[nfft - (lb - 1):] if lb > 1 else np.zeros(0)
    c_lin = np.concatenate([neg, pos])
    lags = np.arange(-(lb - 1), la)
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    c = c_lin / norm
    idx = int(np.argmax(c))
    lag_s = float(lags[idx]) * hop
    return lag_s, float(c[idx])


def refine_local_lag(system_seg: np.ndarray, mic_seg: np.ndarray,
                     sr: int, search_s: float = LOCAL_LAG_SEARCH_S
                     ) -> tuple[float, float]:
    """Return (lag_seconds, normalized_peak_height).

    `normalized_peak_height` is the xcorr peak relative to the std of all xcorr
    values in the search window — a robust SNR-like indicator. >3 is strong.
    Lag is filtered: if peak is weak, returns 0 lag.
    """
    env_sys = rms_envelope(system_seg, sr, ENV_HOP_S)
    env_mic = rms_envelope(mic_seg, sr, ENV_HOP_S)
    if len(env_sys) < 50 or len(env_mic) < 50:
        return 0.0, 0.0
    # Use log-envelope so loud bursts dominate less and broad structure correlates better.
    log_sys = np.log(env_sys + 1e-6)
    log_mic = np.log(env_mic + 1e-6)
    max_lag_frames = int(round(search_s / ENV_HOP_S))
    # Compute the full xcorr trace within the search window so we can score peak prominence.
    a = log_sys.astype(np.float64) - np.mean(log_sys)
    b = log_mic.astype(np.float64) - np.mean(log_mic)
    la, lb = len(a), len(b)
    nfft = 1 << (la + lb - 1 - 1).bit_length()
    A = np.fft.rfft(a, n=nfft)
    B = np.fft.rfft(b, n=nfft)
    c_full = np.fft.irfft(A * np.conj(B), n=nfft)
    pos = c_full[:la]
    neg = c_full[nfft - (lb - 1):] if lb > 1 else np.zeros(0)
    c_lin = np.concatenate([neg, pos])
    lags = np.arange(-(lb - 1), la)
    mask = np.abs(lags) <= max_lag_frames
    c_use = c_lin[mask]
    lags_use = lags[mask]
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    c_use = c_use / norm
    idx = int(np.argmax(c_use))
    peak_val = float(c_use[idx])
    # Prominence: peak relative to sliding-median absolute deviation
    bg = np.delete(c_use, idx)
    if len(bg) > 0:
        bg_std = float(np.std(bg)) + 1e-12
        prom = (peak_val - float(np.median(bg))) / bg_std
    else:
        prom = 0.0
    if prom < 3.0:
        # weak peak — distrust local refinement, fall back to 0 (use prior)
        return 0.0, prom
    return lags_use[idx] * ENV_HOP_S, prom


def gain_match(system_seg: np.ndarray, mic_seg: np.ndarray) -> tuple[np.ndarray, float]:
    sys_peak = float(np.percentile(np.abs(system_seg), PEAK_PCTILE))
    mic_peak = float(np.percentile(np.abs(mic_seg), PEAK_PCTILE))
    if mic_peak < 1e-6:
        return mic_seg, 0.0
    g = sys_peak / mic_peak
    out = np.clip(mic_seg * g, -1.0, 1.0)
    return out, 20.0 * np.log10(max(g, 1e-12))


def find_voice_segment(mic: np.ndarray, sr: int, narration_t: float
                       ) -> tuple[float, float]:
    """Around narration t_offset, find the contiguous mic-active region."""
    a = max(0, int(round((narration_t - VOICE_PRE_S) * sr)))
    b = min(len(mic), int(round((narration_t + VOICE_POST_S) * sr)))
    chunk = mic[a:b]
    env = rms_envelope(chunk, sr, SIL_HOP_S)
    env_db = db(env)
    active = env_db > VOICE_RMS_THRESH_DBFS
    # find longest active run
    best = (0, 0)
    n = len(active)
    j = 0
    while j < n:
        if active[j]:
            k = j
            while k < n and active[k]:
                k += 1
            if (k - j) > (best[1] - best[0]):
                best = (j, k)
            j = k
        else:
            j += 1
    if best[1] - best[0] < int(round(0.5 / SIL_HOP_S)):
        # fall back: ±5 s around narration
        return narration_t - 2.0, narration_t + 5.0
    start_s = a / sr + best[0] * SIL_HOP_S
    end_s = a / sr + best[1] * SIL_HOP_S
    return start_s, end_s


def slugify(label: str) -> str:
    return label.replace(" ", "_").lower()


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(data, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), out, sr, subtype="PCM_16")


def _py(x):
    """Coerce numpy scalars / arrays to native Python types for YAML."""
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def dump_yaml(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(_py(data), sort_keys=False))


def main() -> None:
    session_yaml = yaml.safe_load((SESSION_DIR / "session.yaml").read_text())
    sr_yaml = int(session_yaml["sample_rate_hz"])
    assert sr_yaml == SR, f"unexpected sr in session.yaml: {sr_yaml}"
    start_iso = session_yaml["start_iso"]

    narration: list[dict] = []
    for line in (SESSION_DIR / "narration.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        # normalize bool field — different keys appear in the jsonl
        sysflag = rec.get("system_audio")
        if sysflag is None:
            sysflag = rec.get("has_system_audio", True)
        rec["system_audio"] = bool(sysflag)
        narration.append(rec)
    narration.sort(key=lambda r: float(r["t_offset_s"]))
    print(f"Loaded {len(narration)} narration markers")

    # Read both wavs (PCM_16 -> float32 in [-1, 1])
    print("Reading inmp441.wav ...")
    mic, sr_mic = sf.read(str(SESSION_DIR / "inmp441.wav"), dtype="float32", always_2d=False)
    if mic.ndim > 1:
        mic = mic.mean(axis=1)
    print(f"  mic: {len(mic)} frames @ {sr_mic} Hz ({len(mic)/sr_mic:.1f} s)")
    print("Reading system.wav ...")
    system, sr_sys = sf.read(str(SESSION_DIR / "system.wav"), dtype="float32", always_2d=False)
    if system.ndim > 1:
        system = system.mean(axis=1)
    print(f"  sys: {len(system)} frames @ {sr_sys} Hz ({len(system)/sr_sys:.1f} s)")
    assert sr_mic == SR and sr_sys == SR

    # 1) Global stream lag
    print("\n[1] Estimating global stream lag (envelope xcorr) ...")
    global_lag_s, peak = global_stream_lag_seconds(system, mic, SR)
    print(f"  global_stream_lag_s = {global_lag_s:.3f}  (xcorr peak={peak:.4f})")
    print(f"  interpretation: mic sample 0 corresponds to system sample at "
          f"{int(round(global_lag_s * SR))} ({global_lag_s:.2f} s into system.wav)")

    # 2+3) Per-marker windowed song detection.
    music_markers = [m for m in narration if m["system_audio"]]
    voice_markers = [m for m in narration if not m["system_audio"]]
    print(f"\n[2] Per-marker song detection in system.wav "
          f"({len(music_markers)} music markers) ...")

    sys_dur = len(system) / SR
    matches: list[tuple[Song, dict]] = []
    unmatched_markers: list[dict] = []
    for mi, marker in enumerate(music_markers):
        t = float(marker["t_offset_s"])
        # search window starts ~1s before marker (markers are usually said just before
        # play hits) and runs to ~1s before the next marker (or end of file).
        win_start = max(0.0, t - 1.0)
        if mi + 1 < len(music_markers):
            win_end = float(music_markers[mi + 1]["t_offset_s"]) - 1.0
        else:
            win_end = sys_dur
        song = find_song_in_window(system, SR, win_start, win_end, marker_t_s=t)
        if song is None:
            unmatched_markers.append(marker)
            print(f"  marker '{marker['label']}' @ t={t:.2f}s "
                  f"-> NO SONG in window [{win_start:.2f}, {win_end:.2f}]s")
            continue
        matches.append((song, marker))
        print(f"  marker '{marker['label']:35s}' t={t:8.2f}s "
              f"-> song [{song.start_s:8.2f}, {song.end_s:8.2f}]s "
              f"(dur {song.dur_s:6.2f}s, start_delay {song.start_s - t:+.2f}s)")

    # Sort matches by song start for index assignment AFTER we add the voice segment
    # Voice segment first
    print(f"\n[4] Voice segment: {len(voice_markers)} marker(s)")
    voice_results = []
    for vm in voice_markers:
        t = float(vm["t_offset_s"])
        v_start, v_end = find_voice_segment(mic, SR, t)
        # Note: t and v_start/v_end are in the SYSTEM-side timeline since narration
        # offsets use start_iso ~ system t=0. But we are searching mic.
        # The mic timeline is offset by global_lag_s (mic sample 0 ~ system t=global_lag_s).
        # find_voice_segment indexed mic directly using t as if it were mic-time.
        # Refine: mic-time = system-time - global_lag_s.
        mic_t = t - global_lag_s
        a = max(0, int(round((mic_t - VOICE_PRE_S) * SR)))
        b = min(len(mic), int(round((mic_t + VOICE_POST_S) * SR)))
        chunk = mic[a:b]
        env = rms_envelope(chunk, SR, SIL_HOP_S)
        env_db = db(env)
        active = env_db > VOICE_RMS_THRESH_DBFS
        best = (0, 0)
        n = len(active)
        j = 0
        while j < n:
            if active[j]:
                k = j
                while k < n and active[k]:
                    k += 1
                if (k - j) > (best[1] - best[0]):
                    best = (j, k)
                j = k
            else:
                j += 1
        if best[1] - best[0] < int(round(0.5 / SIL_HOP_S)):
            v_start_mic = mic_t - 2.0
            v_end_mic = mic_t + 5.0
        else:
            v_start_mic = a / SR + best[0] * SIL_HOP_S
            v_end_mic = a / SR + best[1] * SIL_HOP_S
        # convert back to system timeline (=narration t_offset_s) for metadata
        v_start_sys = v_start_mic + global_lag_s
        v_end_sys = v_end_mic + global_lag_s
        print(f"  '{vm['label']}' marker_t={t:.2f}s -> mic-time "
              f"[{v_start_mic:.2f}, {v_end_mic:.2f}]s (dur {v_end_mic - v_start_mic:.2f}s)")
        voice_results.append({
            "marker": vm,
            "v_start_mic": v_start_mic,
            "v_end_mic": v_end_mic,
            "v_start_sys": v_start_sys,
            "v_end_sys": v_end_sys,
        })

    # 5) Per-song extract / align / gain / write
    out_root = SESSION_DIR / "segments"
    out_root.mkdir(exist_ok=True)
    manifest_segs = []
    print("\n[5] Extracting segments ...\n")

    # Index assignment: chronological by t_offset_s of associated marker.
    # voice marker (idx 1) is earliest, so it'll be 01.
    timeline = []
    for vr in voice_results:
        timeline.append(("voice", vr["marker"], vr))
    for song, marker in matches:
        timeline.append(("music", marker, song))
    timeline.sort(key=lambda e: float(e[1]["t_offset_s"]))

    for i, entry in enumerate(timeline, start=1):
        kind, marker, payload = entry
        idx_str = f"{i:02d}"
        label = marker["label"]
        slug = slugify(label)
        seg_dir = out_root / f"{idx_str}_{slug}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        if kind == "voice":
            vr = payload
            mic_a = max(0, int(round(vr["v_start_mic"] * SR)))
            mic_b = min(len(mic), int(round(vr["v_end_mic"] * SR)))
            mic_seg = mic[mic_a:mic_b]
            write_wav(seg_dir / "inmp441.wav", mic_seg, SR)
            meta = {
                "index": idx_str,
                "label": label,
                "mic_position": marker.get("mic_position", "ambient"),
                "system_audio": False,
                "duration_s": round((mic_b - mic_a) / SR, 3),
                "t_offset_s_start": round(vr["v_start_sys"], 3),
                "t_offset_s_end": round(vr["v_end_sys"], 3),
                "mic_lag_ms": None,
                "mic_gain_db": None,
                "narration_marker_t_offset_s": float(marker["t_offset_s"]),
                "notes": "voice segment; system.wav silent — mic-only output.",
            }
            dump_yaml(seg_dir / "metadata.yaml", meta)
            manifest_segs.append({
                "index": idx_str,
                "label": label,
                "dir": seg_dir.name,
                "duration_s": meta["duration_s"],
                "system_audio": False,
            })
            print(f"  [{idx_str}] voice '{label}': mic-only "
                  f"{(mic_b - mic_a)/SR:.2f}s -> {seg_dir.name}/")
            continue

        # music segment
        song = payload
        sys_start = song.start_s
        sys_end = song.end_s

        # Clip the requested span to what the mic stream actually covers
        # (mic-time-end maps to sys-time-end mic_dur + global_lag_s).
        mic_max_sys_time = (len(mic) / SR) + global_lag_s
        mic_min_sys_time = global_lag_s  # mic sample 0 == sys-time global_lag_s
        # leave a 1-second cushion at each edge so we can do lag refinement
        sys_start_clip = max(sys_start, mic_min_sys_time + 1.0)
        sys_end_clip = min(sys_end, mic_max_sys_time - 1.0)
        if sys_end_clip - sys_start_clip < MIN_SONG_S:
            note = (f"mic stream did not cover the song span "
                    f"sys [{sys_start:.1f},{sys_end:.1f}]; "
                    f"mic covers sys-time [{global_lag_s:.1f},{mic_max_sys_time:.1f}].")
            seg_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "index": idx_str,
                "label": label,
                "mic_position": marker.get("mic_position", "ambient"),
                "system_audio": True,
                "duration_s": 0.0,
                "t_offset_s_start": round(sys_start, 3),
                "t_offset_s_end": round(sys_end, 3),
                "mic_lag_ms": None,
                "mic_gain_db": None,
                "narration_marker_t_offset_s": float(marker["t_offset_s"]),
                "global_stream_lag_s": round(global_lag_s, 3),
                "skipped": True,
                "skipped_reason": note,
            }
            dump_yaml(seg_dir / "metadata.yaml", meta)
            manifest_segs.append({
                "index": idx_str,
                "label": label,
                "dir": seg_dir.name,
                "duration_s": 0.0,
                "system_audio": True,
                "skipped": True,
            })
            print(f"  [{idx_str}] '{label}': SKIPPED — {note}")
            continue
        if sys_end_clip < sys_end - 0.05 or sys_start_clip > sys_start + 0.05:
            print(f"  [{idx_str}] '{label}': clipping to mic-covered span "
                  f"[{sys_start_clip:.2f}, {sys_end_clip:.2f}]s "
                  f"(was [{sys_start:.2f}, {sys_end:.2f}])")
        sys_start = sys_start_clip
        sys_end = sys_end_clip

        sys_a = max(0, int(round(sys_start * SR)))
        sys_b = min(len(system), int(round(sys_end * SR)))
        system_seg = system[sys_a:sys_b]

        # mic spans the same wall-clock window: mic_t = sys_t - global_lag_s
        mic_a = max(0, int(round((sys_start - global_lag_s) * SR)))
        mic_b = min(len(mic), int(round((sys_end - global_lag_s) * SR)))
        mic_seg_unaligned = mic[mic_a:mic_b]

        # Refine local lag on the segment envelopes
        local_lag_s, local_peak_z = refine_local_lag(system_seg, mic_seg_unaligned, SR)
        # local_lag_s > 0 means mic lags system by an additional local_lag_s seconds
        # beyond global_lag_s. Re-extract mic with that extra shift.
        mic_a2 = max(0, int(round((sys_start - global_lag_s + local_lag_s) * SR)))
        seg_len = sys_b - sys_a
        mic_b2 = mic_a2 + seg_len
        if mic_b2 > len(mic):
            mic_b2 = len(mic)
            seg_len_adj = mic_b2 - mic_a2
            system_seg = system_seg[:seg_len_adj]
        mic_seg = mic[mic_a2:mic_a2 + len(system_seg)]
        if len(mic_seg) < len(system_seg):
            pad = len(system_seg) - len(mic_seg)
            mic_seg = np.concatenate([mic_seg, np.zeros(pad, dtype=mic_seg.dtype)])

        # Gain match
        mic_seg_g, gain_db = gain_match(system_seg, mic_seg)

        # Write
        write_wav(seg_dir / "system.wav", system_seg, SR)
        write_wav(seg_dir / "inmp441.wav", mic_seg_g, SR)

        meta = {
            "index": idx_str,
            "label": label,
            "mic_position": marker.get("mic_position", "ambient"),
            "system_audio": True,
            "duration_s": round(len(system_seg) / SR, 3),
            "t_offset_s_start": round(sys_start, 3),
            "t_offset_s_end": round(sys_end, 3),
            "mic_lag_ms": round(local_lag_s * 1000.0, 2),
            "mic_gain_db": round(gain_db, 2),
            "narration_marker_t_offset_s": float(marker["t_offset_s"]),
            "global_stream_lag_s": round(global_lag_s, 3),
            "local_xcorr_peak_z": round(local_peak_z, 3),
            "lag_refinement_trusted": local_peak_z >= 3.0,
        }
        dump_yaml(seg_dir / "metadata.yaml", meta)
        manifest_segs.append({
            "index": idx_str,
            "label": label,
            "dir": seg_dir.name,
            "duration_s": meta["duration_s"],
            "system_audio": True,
        })
        trust_tag = "OK" if local_peak_z >= 3.0 else "weak"
        print(f"  [{idx_str}] '{label}': dur={meta['duration_s']:.2f}s, "
              f"local_lag={local_lag_s*1000:+.1f}ms, gain={gain_db:+.2f}dB "
              f"(xcorr z={local_peak_z:.2f} {trust_tag}) -> {seg_dir.name}/")

    # Manifest
    manifest = {
        "session_dir": str(SESSION_DIR),
        "start_iso": start_iso,
        "sample_rate_hz": SR,
        "global_stream_lag_s": round(global_lag_s, 3),
        "segments": manifest_segs,
    }
    dump_yaml(out_root.parent / "manifest.yaml", manifest)

    # Final summary
    print("\n=== Summary ===")
    print(f"Global stream lag (system -> mic): {global_lag_s:.3f} s")
    print(f"Segments produced: {len(manifest_segs)}")
    for s in manifest_segs:
        print(f"  {s['index']} {s['label']:38s} {s['duration_s']:7.2f}s  sys={s['system_audio']}")
    if unmatched_markers:
        print("Unmatched music markers:")
        for m in unmatched_markers:
            print(f"  {m['label']} @ {m['t_offset_s']}s")


if __name__ == "__main__":
    main()
