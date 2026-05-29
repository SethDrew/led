#!/usr/bin/env python3
"""
Dual-port latency probe: listens to the ESP-NOW sniffer and bench-bulbs
simultaneously, printing timestamped lines to measure packet→reaction delay.

Sniffer prints "[NEW] sender ..." on first packet and summary lines with seq numbers.
Bench-bulbs prints "[bloom] ..." status lines when it processes packets.

Usage:
  python latency_probe.py [sniffer_port] [bench_port]
"""

import serial
import time
import sys
import threading
import struct

SNIFFER_PORT = '/dev/cu.usbmodem1121201'
BENCH_PORT = '/dev/cu.usbserial-0001'
SNIFFER_BAUD = 460800
BENCH_BAUD = 460800


def monitor_port(port_path, baud, label, stop_event):
    try:
        s = serial.Serial(port_path, baud, timeout=0.05)
    except Exception as e:
        print(f"[{label}] FAILED to open {port_path}: {e}")
        return

    t0 = time.monotonic()
    buf = b''
    while not stop_event.is_set():
        chunk = s.read(256)
        if not chunk:
            continue
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            text = line.decode('utf-8', errors='replace').strip()
            if not text:
                continue
            # Filter to interesting lines
            show = False
            if label == 'SNIFF':
                # Show binary frame arrivals (0xA5 0x5A header) or text lines with sender info
                if '[NEW]' in text or 'sender' in text or 'seq=' in text:
                    show = True
                # Also count raw binary frames
            if label == 'BENCH':
                if 'bloom' in text.lower() or 'espnow' in text.lower() or 'rate=' in text or 'PX' in text:
                    show = True

            if show:
                elapsed = time.monotonic() - t0
                print(f"[{elapsed:8.3f}] {label:5s} | {text[:120]}")

    s.close()


def monitor_sniffer_binary(port_path, baud, stop_event):
    """Monitor sniffer for binary frames (0xA5 0x5A) and print decoded v1 packets."""
    try:
        s = serial.Serial(port_path, baud, timeout=0.05)
    except Exception as e:
        print(f"[SNIFF-BIN] FAILED: {e}")
        return

    t0 = time.monotonic()
    buf = b''
    pkt_count = 0
    last_print = 0

    while not stop_event.is_set():
        chunk = s.read(512)
        if not chunk:
            continue
        buf += chunk

        # Scan for binary frames: [0xA5][0x5A][LEN][MAC:6][PAYLOAD:LEN-6][XOR8]
        while len(buf) >= 3:
            idx = buf.find(b'\xa5\x5a')
            if idx < 0:
                buf = buf[-1:]  # keep last byte in case it's 0xA5
                break
            if idx > 0:
                buf = buf[idx:]
            if len(buf) < 3:
                break
            frame_len = buf[2]  # LEN = MAC(6) + payload
            total = 3 + frame_len + 1  # header(3) + data(frame_len) + xor(1)
            if len(buf) < total:
                break

            # Verify XOR8
            xor8 = 0
            for i in range(2, 3 + frame_len):
                xor8 ^= buf[i]
            if xor8 != buf[3 + frame_len]:
                buf = buf[2:]  # bad frame, skip sync bytes
                continue

            mac = buf[3:9]
            payload = buf[9:3 + frame_len]
            buf = buf[total:]

            pkt_count += 1
            now = time.monotonic()
            elapsed = now - t0

            if len(payload) == 16:
                seq = struct.unpack_from('<H', payload, 0)[0]
                gmag_max = payload[13]
                gmag_dps = (gmag_max / 255.0) ** 2 * 57000 / 32.8
                # Print every packet during motion, every 25th when quiet
                if gmag_dps > 10 or pkt_count % 25 == 0 or now - last_print > 1.0:
                    print(f"[{elapsed:8.3f}] SNIFF | pkt#{pkt_count} seq={seq} gmag={gmag_dps:.0f}dps MAC=..{mac[4]:02X}:{mac[5]:02X}")
                    last_print = now

    s.close()


def main():
    sniffer_port = sys.argv[1] if len(sys.argv) > 1 else SNIFFER_PORT
    bench_port = sys.argv[2] if len(sys.argv) > 2 else BENCH_PORT

    print(f"Sniffer: {sniffer_port}")
    print(f"Bench:   {bench_port}")
    print(f"Listening... (Ctrl-C to stop)\n")

    stop = threading.Event()

    threads = [
        threading.Thread(target=monitor_sniffer_binary, args=(sniffer_port, SNIFFER_BAUD, stop), daemon=True),
        threading.Thread(target=monitor_port, args=(bench_port, BENCH_BAUD, 'BENCH', stop), daemon=True),
    ]

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop.set()
        for t in threads:
            t.join(timeout=2)


if __name__ == '__main__':
    main()
