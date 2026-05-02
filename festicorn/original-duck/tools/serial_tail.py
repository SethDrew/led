#!/usr/bin/env python3
"""Tail serial output from duck + receiver ESP32-C3 boards to log files.

Long-lived daemon: per-board thread, auto-reconnect on USB-CDC drop
(common when the duck enters light-sleep), timestamped lines, fresh
log files each run.
"""

import os
import signal
import sys
import threading
import time
from datetime import datetime

import serial

BOARDS = [
    {"name": "duck",     "port": "/dev/cu.usbmodem1114401", "log": "/tmp/duck.log"},
    {"name": "receiver", "port": "/dev/cu.usbmodem1114201", "log": "/tmp/receiver.log"},
]
BAUD = 460800

stop_event = threading.Event()


def ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def tail_board(name: str, port: str, log_path: str) -> None:
    # Fresh log each run.
    log_f = open(log_path, "w", buffering=1)  # line-buffered
    log_f.write(f"{ts()} --- serial_tail start: {name} {port} @ {BAUD} ---\n")

    backoff = 1.0
    ser = None

    while not stop_event.is_set():
        # Open phase
        if ser is None:
            try:
                if not os.path.exists(port):
                    raise serial.SerialException(f"port {port} not present")
                # ESP32-C3 USB-CDC: DTR=reset, RTS=GPIO9(BOOT). Asserting both
                # at open time forces download mode. Configure FIRST, then open.
                ser = serial.Serial()
                ser.port = port
                ser.baudrate = BAUD
                ser.timeout = 1
                ser.dtr = False
                ser.rts = False
                ser.open()
                log_f.write(f"{ts()} --- reconnected ---\n")
                backoff = 1.0
            except (serial.SerialException, OSError) as e:
                log_f.write(f"{ts()} --- waiting for port ({e}) ---\n")
                # Wait with backoff but stay responsive to stop_event.
                stop_event.wait(timeout=backoff)
                backoff = min(backoff * 1.5, 5.0)
                continue

        # Read phase
        try:
            line = ser.readline()
            if not line:
                continue  # timeout, loop again
            try:
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                text = repr(line)
            log_f.write(f"{ts()} {text}\n")
        except (serial.SerialException, OSError) as e:
            log_f.write(f"{ts()} --- disconnected ({e}) ---\n")
            try:
                ser.close()
            except Exception:
                pass
            ser = None
            stop_event.wait(timeout=backoff)
            backoff = min(backoff * 1.5, 5.0)

    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    log_f.write(f"{ts()} --- serial_tail stop: {name} ---\n")
    log_f.close()


def handle_signal(signum, frame):
    stop_event.set()


def main() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    threads = []
    for b in BOARDS:
        t = threading.Thread(
            target=tail_board,
            args=(b["name"], b["port"], b["log"]),
            name=f"tail-{b['name']}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Wait for stop, but don't block on join (threads are daemon).
    while not stop_event.is_set():
        time.sleep(0.5)

    for t in threads:
        t.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
