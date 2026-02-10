#!/usr/bin/env python3
"""
Find the LED Tree controller by serial port scan

Connects to each USB serial port and reads the startup message
to identify which one is the tree controller.
"""

import serial
import serial.tools.list_ports
import time
import sys

def find_tree():
    """Scan USB serial ports to find the tree controller"""
    ports = [p.device for p in serial.tools.list_ports.comports() if 'usbserial' in p.device]

    if not ports:
        print("No USB serial devices found")
        return None

    print(f"Scanning {len(ports)} port(s)...")

    for port in ports:
        try:
            print(f"  Checking {port}...", end=" ")
            # Open at low speed (115200) since we don't know what's running
            ser = serial.Serial(port, 115200, timeout=0.5)
            time.sleep(0.1)  # Let it settle

            # Trigger a reset by toggling DTR
            ser.setDTR(False)
            time.sleep(0.1)
            ser.setDTR(True)
            time.sleep(2)  # Wait for bootloader and startup

            # Read any startup messages
            output = ""
            start_time = time.time()
            while time.time() - start_time < 1:
                if ser.in_waiting:
                    chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    output += chunk
                    time.sleep(0.1)

            ser.close()

            # Check if this is the tree
            if "LED TREE" in output or "Tree Effects" in output or "Tree nodes" in output:
                print("✓ TREE FOUND")
                return port
            else:
                print("(not tree)")

        except Exception as e:
            print(f"(error: {e})")
            continue

    return None

if __name__ == "__main__":
    port = find_tree()
    if port:
        print(f"\n✓ Tree controller: {port}")
        sys.exit(0)
    else:
        print("\n✗ Tree controller not found")
        sys.exit(1)
