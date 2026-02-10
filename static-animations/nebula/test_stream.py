#!/usr/bin/env python3
"""
A/B test: solid green then nebula, to find where it breaks
"""

import serial
import time
import numpy as np
from nebula_stream import NebulaEffect

PORT = '/dev/cu.usbserial-1230'
BAUD = 1000000
NUM_LEDS = 150

print(f"Opening {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2.5)

while ser.in_waiting:
    print(f"  Arduino: {ser.readline().decode('utf-8', errors='ignore').strip()}")

# Build green packet the EXACT same way that worked before
green_packet = bytearray([0xFF, 0xAA])
for i in range(NUM_LEDS):
    green_packet.extend([0, 100, 0])

effect = NebulaEffect(NUM_LEDS, speed_multiplier=0.4, tail_length=30.0, max_orbs=5)

print("\n=== Phase 1: Solid green (should work) ===")
for i in range(90):
    ser.write(green_packet)
    if i % 30 == 0:
        print(f"  Green frame {i}")
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line: print(f"  Arduino: {line}")
    time.sleep(1/30)

print("\n=== Phase 2: Nebula via numpy tobytes ===")
for i in range(90):
    frame = effect.update(1/30)
    frame = (frame * 0.5).astype(np.uint8)
    packet = bytearray([0xFF, 0xAA])
    packet.extend(frame.flatten().tobytes())
    ser.write(packet)
    if i % 30 == 0:
        print(f"  Nebula frame {i}: max={frame.max()} dtype={frame.dtype} len={len(packet)}")
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line: print(f"  Arduino: {line}")
    time.sleep(1/30)

print("\n=== Phase 3: Green again (should work) ===")
for i in range(90):
    ser.write(green_packet)
    if i % 30 == 0:
        print(f"  Green frame {i}")
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line: print(f"  Arduino: {line}")
    time.sleep(1/30)

time.sleep(0.5)
while ser.in_waiting:
    print(f"  Arduino: {ser.readline().decode('utf-8', errors='ignore').strip()}")
ser.close()
print("Done!")
