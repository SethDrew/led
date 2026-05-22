#!/usr/bin/env python3
"""Measure the firmware's onset detector against syllable-rate RMS peaks
in real speech recordings.

Replays a binary SensorPacket recording (15-byte packets at ~25 Hz) and
runs the same sparkle-burst onset/trigger logic as effects.cpp:
  delta     = abs(rms - prevRms)
  deltaPeak = max(delta, deltaPeak * 0.998^(dt*25))   # dt-normalized
  onset     = delta / deltaPeak
  envelope  = EMA(energy)  attack 30ms, decay 400ms
  threshold = max(0.15, 0.4 - envelope*0.3)
  cooldown  = max(0.050, 0.150 - envelope*0.10)

Independently finds RMS peaks (local maxima above noise floor) and
classifies each trigger as hit / miss / false.
"""
import struct
import sys
import math
from pathlib import Path

PACKET_SIZE = 15
PACKET_FMT  = "<hhhhhhHB"        # 6×int16, uint16, uint8
PACKET_RATE = 25.0               # Hz, nominal
DT          = 1.0 / PACKET_RATE

# Mirror effects.cpp constants
RMS_FLOOR_INIT       = 700.0
RMS_CEILING          = 5000.0
RMS_PEAK_DECAY       = 0.9999
RMS_FLOOR_EMA_UP     = 0.002
RMS_FLOOR_EMA_DOWN   = 0.0005
RMS_FLOOR_HEADROOM   = 1.3
# Peak detector
PEAK_HEADROOM        = 1.15      # rms must exceed effectiveFloor*this
PEAK_MIN_PROMINENCE  = 80.0      # rms above local min
PEAK_MATCH_WINDOW    = 1         # +/- packets


def read_packets(path):
    data = Path(path).read_bytes()
    n = len(data) // PACKET_SIZE
    pkts = []
    for i in range(n):
        ax, ay, az, gx, gy, gz, rms, mic = struct.unpack(
            PACKET_FMT, data[i*PACKET_SIZE:(i+1)*PACKET_SIZE]
        )
        pkts.append(rms)
    return pkts


def run_detector(rms_seq, dt=DT):
    """Replay firmware onset + sparkle-burst trigger logic (log-domain detector).
    Returns parallel arrays: energy, onset, threshold, triggers (bool).
    """
    frrCeiling = RMS_CEILING
    frrFloor   = RMS_FLOOR_INIT
    envelope   = 0.0
    cooldown   = 0.0

    # Log-domain fast/slow EMA state
    logFast = None
    logSlow = None

    energy_arr, onset_arr, thr_arr, trig_arr, floor_arr = [], [], [], [], []

    for rms in rms_seq:
        rmsf = float(rms)

        # --- energy (computeEnergy) ---
        frrCeiling = max(RMS_CEILING, frrCeiling * RMS_PEAK_DECAY)
        if rmsf > frrCeiling: frrCeiling = rmsf
        alpha = RMS_FLOOR_EMA_UP if rmsf > frrFloor else RMS_FLOOR_EMA_DOWN
        frrFloor += alpha * (rmsf - frrFloor)
        frrFloor = max(frrFloor, 100.0)
        effFloor = frrFloor * RMS_FLOOR_HEADROOM
        if rmsf < effFloor:
            energy = 0.0
        else:
            db = 20.0 * math.log10(rmsf / effFloor)
            dbRange = 20.0 * math.log10(frrCeiling / effFloor)
            if dbRange < 1.0: dbRange = 1.0
            energy = max(0.0, min(1.0, db / dbRange))

        # --- onset (log-domain fast/slow EMA) ---
        logRms = math.log2(rmsf + 1.0)
        if logFast is None:
            logFast = logRms
            logSlow = logRms
            onset = 0.0
        else:
            fastA = min(1.0, dt / 0.030) if logRms > logFast else min(1.0, dt / 0.080)
            logFast += fastA * (logRms - logFast)
            slowA = min(1.0, dt / 0.800) if logRms > logSlow else min(1.0, dt / 2.0)
            logSlow += slowA * (logRms - logSlow)
            onset = max(0.0, logFast - logSlow)

        # --- sparkle-burst envelope + trigger ---
        isSilent = energy < 0.001
        attackA = min(1.0, dt / 0.030)
        decayA  = min(1.0, dt / 0.400)
        if energy > envelope:
            envelope += attackA * (energy - envelope)
        else:
            envelope += decayA  * (energy - envelope)
        cooldown = max(0.0, cooldown - dt)
        thr = max(0.3, 0.8 - envelope * 0.5)
        fired = False
        if onset > thr and cooldown <= 0.0 and not isSilent:
            cooldown = max(0.040, 0.120 - envelope * 0.08)
            fired = True

        energy_arr.append(energy)
        onset_arr.append(onset)
        thr_arr.append(thr)
        trig_arr.append(fired)
        floor_arr.append(effFloor)

    return energy_arr, onset_arr, thr_arr, trig_arr, floor_arr


def find_rms_peaks(rms_seq, floor_seq):
    """Local maxima where rms > floor*PEAK_HEADROOM and prominence above
    neighbors > PEAK_MIN_PROMINENCE."""
    peaks = []
    for i in range(1, len(rms_seq) - 1):
        r = float(rms_seq[i])
        if r <= floor_seq[i] * PEAK_HEADROOM:
            continue
        if not (r >= rms_seq[i-1] and r >= rms_seq[i+1]):
            continue
        # prominence vs immediate neighbors
        prom = r - min(rms_seq[i-1], rms_seq[i+1])
        if prom < PEAK_MIN_PROMINENCE:
            continue
        peaks.append(i)
    return peaks


def classify(peaks, triggers):
    trig_idx = [i for i, t in enumerate(triggers) if t]
    matched_trig = set()
    hits, misses = 0, 0
    miss_idx = []
    for p in peaks:
        found = None
        for ti in trig_idx:
            if abs(ti - p) <= PEAK_MATCH_WINDOW and ti not in matched_trig:
                found = ti
                break
        if found is not None:
            hits += 1
            matched_trig.add(found)
        else:
            misses += 1
            miss_idx.append(p)
    false_idx = [ti for ti in trig_idx if ti not in matched_trig]
    return hits, misses, len(false_idx), miss_idx, false_idx, trig_idx


def render_timeline(rms_seq, onset_arr, thr_arr, trig_arr, peaks, floor_seq):
    peak_set = set(peaks)
    print(f"  i  t(s)   rms  floor  onset  thr  marks")
    print(f"  -- -----  ----  -----  -----  ----  -----")
    for i, rms in enumerate(rms_seq):
        marks = ""
        if i in peak_set:  marks += "P"
        if trig_arr[i]:    marks += "T"
        if not marks:      marks = " "
        bar_len = int(min(1.0, rms / 5000.0) * 30)
        bar = "#" * bar_len
        t = i * DT
        print(f"  {i:3d} {t:5.2f}  {rms:4d}  {int(floor_seq[i]):5d}  "
              f"{onset_arr[i]:.2f}  {thr_arr[i]:.2f}  {marks:2s}  {bar}")


def analyze(path, show_timeline=True):
    print(f"\n=== {Path(path).name} ===")
    rms_seq = read_packets(path)
    print(f"packets: {len(rms_seq)}  duration: {len(rms_seq)*DT:.2f}s")
    energy, onset, thr, trig, floor = run_detector(rms_seq)
    peaks = find_rms_peaks(rms_seq, floor)
    hits, misses, false_n, miss_idx, false_idx, trig_idx = classify(peaks, trig)

    print(f"RMS range:    {min(rms_seq)} .. {max(rms_seq)}")
    print(f"floor final:  {int(floor[-1])}  (eff-headroom included)")
    print(f"peaks found:  {len(peaks)}")
    print(f"triggers:     {len(trig_idx)}")
    print(f"  hits:       {hits}  ({100*hits/max(1,len(peaks)):.0f}% of peaks)")
    print(f"  misses:     {misses}  (peaks with NO trigger within +/-1)")
    print(f"  false:      {false_n}  (triggers not near any peak)")
    if miss_idx:
        print(f"  miss frames: {miss_idx[:30]}{' ...' if len(miss_idx)>30 else ''}")
    if false_idx:
        print(f"  false frames:{false_idx[:30]}{' ...' if len(false_idx)>30 else ''}")
    if show_timeline:
        print()
        render_timeline(rms_seq, onset, thr, trig, peaks, floor)


if __name__ == "__main__":
    base = "/Users/sethdrew/Documents/projects/led/library/test-vectors/phone-sensor-profiles"
    files = [
        f"{base}/s24_home_close_talk_10s.bin",
        f"{base}/s24_home_far_talk_10s.bin",
    ]
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    for f in files:
        analyze(f, show_timeline=True)
