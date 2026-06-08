#!/usr/bin/env python3
"""Dither A/B test controller. Reads serial, accepts commands via /tmp/dither_cmd pipe."""
import serial
import time
import sys
import os
import select

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200
CMD_PIPE = "/tmp/dither_cmd"

ser = serial.Serial(PORT, BAUD, timeout=0.1)
ser.dtr = False
time.sleep(0.1)
ser.dtr = True
time.sleep(2)
ser.reset_input_buffer()

print("READY", flush=True)

cmd_fd = os.open(CMD_PIPE, os.O_RDONLY | os.O_NONBLOCK)

while True:
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='replace').strip()
        if line:
            print(line, flush=True)

    try:
        data = os.read(cmd_fd, 64)
        if data:
            for ch in data.decode().strip():
                if ch in ('n', 'p', 'q'):
                    ser.write(ch.encode())
                    time.sleep(0.05)
    except BlockingIOError:
        pass

    time.sleep(0.05)
