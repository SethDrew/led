#!/usr/bin/env python3
import json
import sys
import time

import serial

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbserial-2120"
out = sys.argv[2] if len(sys.argv) > 2 else "capture.jsonl"

s = serial.Serial(port, 230400, timeout=1)
n = 0
with open(out, "w") as f:
    print(f"recording {port} -> {out}", flush=True)
    try:
        while True:
            raw = s.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            d["host_ts"] = time.time()
            f.write(json.dumps(d) + "\n")
            n += 1
            if n % 100 == 0:
                f.flush()
                print(f"{n} samples", flush=True)
    except KeyboardInterrupt:
        pass
print(f"done: {n} samples", flush=True)
