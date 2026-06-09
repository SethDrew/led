#!/usr/bin/env python3
"""Capture knob sweep data from dashboard API. Detects position transitions, groups by position."""
import json
import time
import urllib.request
import sys
from collections import defaultdict
import statistics

URL = "http://localhost:8080/api/state"
POLL_HZ = 4

samples = []
print("Capturing... sweep knobs at your pace. Ctrl-C to stop and summarize.")
print()

try:
    prev = None
    while True:
        try:
            r = urllib.request.urlopen(URL, timeout=1)
            d = json.loads(r.read())
        except Exception:
            time.sleep(1 / POLL_HZ)
            continue

        t = time.time()
        h, te, o = d["hundreds"], d["tens"], d["ones"]
        do, dm, di = d["decOuter"], d["decMid"], d["decInner"]
        key = (h, te, o, do, dm, di)

        if prev != key:
            if prev is not None:
                print(f"  -> h={h} t={te} o={o} dec={do}/{dm}/{di}")
            prev = key

        samples.append({
            "t": t,
            "hundreds": h, "tens": te, "ones": o,
            "decOuter": do, "decMid": dm, "decInner": di,
            "rawH": d["rawH"], "rawT": d["rawT"], "rawO": d["rawO"],
            "adsT": d["adsT"], "adsO": d["adsO"], "adsRef": d["adsRef"],
        })
        time.sleep(1 / POLL_HZ)

except KeyboardInterrupt:
    pass

print(f"\n{'='*60}")
print(f"Captured {len(samples)} samples")
print()

for knob, pos_key, raw_key in [
    ("HUNDREDS", "hundreds", "rawH"),
    ("TENS", "tens", "rawT"),
    ("ONES", "ones", "rawO"),
]:
    groups = defaultdict(list)
    for s in samples:
        groups[s[pos_key]].append(s[raw_key])

    if len(groups) <= 1:
        continue

    print(f"=== {knob} (native ESP32 ADC) ===")
    print(f"{'Pos':<5} {'Median':>7} {'Min':>7} {'Max':>7} {'Spread':>7} {'N':>5}")
    for pos in sorted(groups):
        vals = groups[pos]
        med = int(statistics.median(vals))
        print(f"{pos:<5} {med:>7} {min(vals):>7} {max(vals):>7} {max(vals)-min(vals):>7} {len(vals):>5}")
    print()

for knob, pos_key, raw_key in [
    ("DECADE OUTER", "decOuter", "adsT"),
    ("DECADE INNER", "decInner", "adsO"),
]:
    groups = defaultdict(list)
    for s in samples:
        groups[s[pos_key]].append(s[raw_key])

    if len(groups) <= 1:
        continue

    print(f"=== {knob} (ADS1115) ===")
    print(f"{'Pos':<5} {'Median':>7} {'Min':>7} {'Max':>7} {'Spread':>7} {'N':>5}")
    for pos in sorted(groups):
        vals = groups[pos]
        med = int(statistics.median(vals))
        print(f"{pos:<5} {med:>7} {min(vals):>7} {max(vals):>7} {max(vals)-min(vals):>7} {len(vals):>5}")
    print()

# adsRef summary
refs = [s["adsRef"] for s in samples]
print(f"=== ADS REF ===")
print(f"Median: {int(statistics.median(refs))}  Range: {min(refs)}-{max(refs)}  Spread: {max(refs)-min(refs)}")
