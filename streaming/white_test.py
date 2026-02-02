#!/usr/bin/env python3
"""
White Test for LED Current Measurement

Sends solid white at a specified brightness to all LEDs.
Perfect for measuring current draw with an amp meter.

Usage:
    python white_test.py [--port PORT] [--leds NUM] [--brightness PERCENT]
"""

import serial
import time
import numpy as np
import argparse

# Protocol constants (same as stream_controller)
START_BYTE_1 = 0xFF
START_BYTE_2 = 0xAA

class LEDStreamer:
    """Manages serial connection and frame streaming"""
    def __init__(self, port: str, num_leds: int, baud_rate: int = 1000000):
        self.num_leds = num_leds
        self.ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset

        # Read startup messages
        while self.ser.in_waiting:
            print(self.ser.readline().decode('utf-8', errors='ignore').strip())

    def send_frame(self, frame: np.ndarray):
        """Send RGB frame to Arduino"""
        # Build packet: [START1] [START2] [RGB data...]
        packet = bytearray([START_BYTE_1, START_BYTE_2])
        packet.extend(frame.flatten().tobytes())

        self.ser.write(packet)

    def close(self):
        self.ser.close()

def main():
    parser = argparse.ArgumentParser(description='White test for LED current measurement')
    parser.add_argument('--port', default='/dev/cu.usbserial-1230', help='Serial port')
    parser.add_argument('--leds', type=int, default=150, help='Number of LEDs')
    parser.add_argument('--brightness', type=float, default=60.0, help='Brightness percentage (0-100)')
    args = parser.parse_args()

    # Convert brightness percentage to 0-255 value
    brightness_value = int((args.brightness / 100.0) * 255)
    brightness_value = max(0, min(255, brightness_value))  # Clamp to 0-255

    print(f"LED White Test")
    print(f"  Port: {args.port}")
    print(f"  LEDs: {args.leds}")
    print(f"  Brightness: {args.brightness}% ({brightness_value}/255)")
    print(f"  Color: Full White (R={brightness_value}, G={brightness_value}, B={brightness_value})")
    print()

    # Initialize
    streamer = LEDStreamer(args.port, args.leds)

    # Create frame with solid white at specified brightness
    frame = np.full((args.leds, 3), brightness_value, dtype=np.uint8)

    print("Sending white test pattern...")
    print("Press Ctrl+C to stop")
    print()

    try:
        frame_count = 0
        start_time = time.time()

        while True:
            # Send the same frame repeatedly
            streamer.send_frame(frame)
            frame_count += 1

            # Print stats every 60 frames
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frames sent: {frame_count}, FPS: {fps:.1f}")

            # Small delay to avoid overwhelming the serial port
            time.sleep(0.016)  # ~60 FPS

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        streamer.close()
        print("Done!")

if __name__ == '__main__':
    main()
