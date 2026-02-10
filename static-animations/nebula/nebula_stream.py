#!/usr/bin/env python3
"""
LED Streaming Controller

Calculates LED effect frames on the computer and streams to Arduino via serial.
Much faster computation than Arduino - enables complex effects at high FPS.

Usage:
    python stream_controller.py [--port PORT] [--leds NUM] [--fps FPS]

Effects available:
    - Nebula: The breathing wave + glowing orbs effect
    - More effects can be added easily!
"""

import serial
import time
import numpy as np
import argparse
from typing import Tuple

# Protocol constants
START_BYTE_1 = 0xFF
START_BYTE_2 = 0xAA

class NebulaEffect:
    """
    Nebula effect - breathing waves with glowing orbs
    (Python version of the Arduino effect, but much faster!)
    """
    def __init__(self, num_pixels: int, speed_multiplier: float = 1.0, tail_length: float = 7.0, max_orbs: int = 5, min_lifetime: int = 200, max_lifetime: int = 300):
        self.num_pixels = num_pixels
        self.elapsed_time = 0.0  # Time in seconds (deterministic!)
        self.last_update = None
        self.speed = speed_multiplier  # Global speed control!

        # Background parameters (scaled to 0-255 range)
        self.BREATH_FREQUENCY = 0.0105 * speed_multiplier  # 30% of original (0.035 * 0.3)
        self.BREATH_CENTER = 51
        self.BREATH_AMPLITUDE = 38
        self.SPATIAL_AMPLITUDE = 51
        self.SPATIAL_SPEED = 0.006 * speed_multiplier  # 30% of original (0.02 * 0.3)
        self.BACKGROUND_MAX = 153

        # Orbs with decay buffer (like Arduino version)
        self.max_orbs = max_orbs
        self.min_lifetime = min_lifetime
        self.max_lifetime = max_lifetime
        self.orb_size = tail_length  # Controls decay/trail length (higher = longer trails!)
        self.orb_base_speed = 0.45  # Base orb velocity
        self.orbs = []
        self.orb_brightness = np.zeros(num_pixels, dtype=np.float32)  # Decay buffer!

    def update(self, dt: float) -> np.ndarray:
        """
        Calculate and return next frame (RGB values 0-255)

        Args:
            dt: Delta time since last frame (in seconds) - makes animation speed deterministic!
        """
        self.elapsed_time += dt

        # Initialize frame
        frame = np.zeros((self.num_pixels, 3), dtype=np.uint8)

        # === BACKGROUND: Breathing waves ===
        # Use elapsed time for deterministic speed (independent of FPS!)
        t = self.elapsed_time * 60.0  # Scale to match original frame-based timing
        breathing = self.BREATH_CENTER + self.BREATH_AMPLITUDE * np.sin(t * self.BREATH_FREQUENCY)

        # Spatial wave for each LED
        positions = np.arange(self.num_pixels) / self.num_pixels
        phases = positions + t * self.SPATIAL_SPEED
        spatial = self.SPATIAL_AMPLITUDE * (0.5 + 0.5 * np.cos(2.0 * np.pi * phases))

        # Combine and clamp
        bg_brightness = np.clip(breathing + spatial, 0, self.BACKGROUND_MAX).astype(np.uint8)

        # Color variation (blue to magenta)
        color_phases = positions * 2.0 * np.pi + t * 0.009  # 30% of original (0.03 * 0.3)
        color_shift = 0.5 + 0.5 * np.sin(color_phases)

        bg_r = (20 + color_shift * 235).astype(np.uint8)
        bg_g = (30 - color_shift * 20).astype(np.uint8)
        bg_b = (255 - color_shift * 125).astype(np.uint8)

        # Apply brightness to colors (proper multiplication to avoid truncation)
        frame[:, 0] = ((bg_r.astype(np.uint16) * bg_brightness) >> 8).astype(np.uint8)
        frame[:, 1] = ((bg_g.astype(np.uint16) * bg_brightness) >> 8).astype(np.uint8)
        frame[:, 2] = ((bg_b.astype(np.uint16) * bg_brightness) >> 8).astype(np.uint8)

        # === FOREGROUND: Glowing orbs with decay trails ===

        # DECAY BUFFER (properly time-based using exponential decay)
        # decay_factor per frame at 60 FPS = (1 - 1/orb_size)
        # Convert to per-second rate, then apply based on actual dt
        decay_per_frame_at_60fps = 1.0 - (1.0 / self.orb_size)
        decay_per_second = decay_per_frame_at_60fps ** 60.0
        time_based_decay = decay_per_second ** dt
        self.orb_brightness *= time_based_decay
        self.orb_brightness[self.orb_brightness < 0.01] = 0.0

        # Spawn orbs
        if len(self.orbs) < self.max_orbs and np.random.rand() < 0.03:
            self.orbs.append({
                'position': np.random.randint(0, self.num_pixels),
                'velocity': np.random.choice([-1, 1]) * self.orb_base_speed * self.speed * np.random.uniform(0.7, 1.3),  # 70-130% speed variation
                'age': 0,
                'lifetime': np.random.randint(self.min_lifetime, self.max_lifetime)
            })

        # Update and render orbs
        orbs_to_remove = []
        for orb in self.orbs:
            orb['age'] += 1
            orb['position'] += orb['velocity']
            orb['position'] %= self.num_pixels

            if orb['age'] >= orb['lifetime']:
                orbs_to_remove.append(orb)
                continue

            # Calculate lifecycle brightness (smoothstep fade in/out)
            lifecycle = orb['age'] / orb['lifetime']
            if lifecycle < 0.4:
                t_fade = lifecycle / 0.4
                brightness = t_fade * t_fade * (3.0 - 2.0 * t_fade)
            elif lifecycle > 0.6:
                t_fade = (1.0 - lifecycle) / 0.4
                brightness = t_fade * t_fade * (3.0 - 2.0 * t_fade)
            else:
                brightness = 1.0

            # Add brightness to decay buffer with sub-pixel interpolation (smooth fade-forward)
            position = orb['position']
            pixel_base = int(position)
            pixel_next = (pixel_base + 1) % self.num_pixels
            fraction = position - pixel_base  # Fractional part (0.0 to 1.0)

            # Split brightness between current and next LED based on position
            # This creates smooth fade-forward instead of discrete jumps
            if 0 <= pixel_base < self.num_pixels:
                self.orb_brightness[pixel_base] = min(1.0, self.orb_brightness[pixel_base] + brightness * 0.6 * (1.0 - fraction))
            if 0 <= pixel_next < self.num_pixels:
                self.orb_brightness[pixel_next] = min(1.0, self.orb_brightness[pixel_next] + brightness * 0.6 * fraction)

        # Remove dead orbs
        for orb in orbs_to_remove:
            self.orbs.remove(orb)

        # Render orb layer (warm white) from decay buffer
        orb_mask = self.orb_brightness > 0.01
        star_r = (255 * self.orb_brightness[orb_mask]).astype(np.uint16)
        star_g = (240 * self.orb_brightness[orb_mask]).astype(np.uint16)
        star_b = (200 * self.orb_brightness[orb_mask]).astype(np.uint16)

        frame[orb_mask, 0] = np.clip(frame[orb_mask, 0].astype(np.uint16) + star_r, 0, 255).astype(np.uint8)
        frame[orb_mask, 1] = np.clip(frame[orb_mask, 1].astype(np.uint16) + star_g, 0, 255).astype(np.uint8)
        frame[orb_mask, 2] = np.clip(frame[orb_mask, 2].astype(np.uint16) + star_b, 0, 255).astype(np.uint8)

        return frame

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
    parser = argparse.ArgumentParser(description='Stream LED effects to Arduino')
    parser.add_argument('--port', default='/dev/cu.usbserial-1230', help='Serial port')
    parser.add_argument('--leds', type=int, default=150, help='Number of LEDs')
    parser.add_argument('--fps', type=int, default=60, help='Target frames per second')
    parser.add_argument('--brightness', type=float, default=0.5, help='Brightness (0.0-1.0)')
    parser.add_argument('--speed', type=float, default=.4, help='Animation speed multiplier (0.5=half speed, 2.0=double speed)')
    parser.add_argument('--tail-length', type=float, default=30.0, help='Orb tail length (7=short, 15=medium, 30=long)')
    parser.add_argument('--orbs', type=int, default=5, help='Maximum number of orbs (1-10)')
    parser.add_argument('--min-lifetime', type=int, default=200, help='Minimum orb lifetime in frames (default: 200)')
    parser.add_argument('--max-lifetime', type=int, default=300, help='Maximum orb lifetime in frames (default: 300)')
    args = parser.parse_args()

    print(f"LED Stream Controller")
    print(f"  Port: {args.port}")
    print(f"  LEDs: {args.leds}")
    print(f"  Target FPS: {args.fps}")
    print(f"  Brightness: {args.brightness * 100}%")
    print(f"  Speed: {args.speed}x")
    print(f"  Tail Length: {args.tail_length}")
    print(f"  Max Orbs: {args.orbs}")
    print(f"  Lifetime: {args.min_lifetime}-{args.max_lifetime} frames ({args.min_lifetime/args.fps:.1f}-{args.max_lifetime/args.fps:.1f}s)")
    print()

    # Initialize
    streamer = LEDStreamer(args.port, args.leds)
    effect = NebulaEffect(args.leds, speed_multiplier=args.speed, tail_length=args.tail_length, max_orbs=args.orbs, min_lifetime=args.min_lifetime, max_lifetime=args.max_lifetime)

    frame_time = 1.0 / args.fps

    print("Streaming... Press Ctrl+C to stop")

    try:
        frame_count = 0
        start_time = time.time()
        last_frame_time = start_time

        while True:
            loop_start = time.time()

            # Calculate delta time (time since last frame)
            if frame_count == 0:
                dt = frame_time  # First frame
            else:
                dt = time.time() - last_frame_time
            last_frame_time = time.time()

            # Calculate frame (dt makes animation speed deterministic!)
            frame = effect.update(dt)

            # Apply brightness
            frame = (frame * args.brightness).astype(np.uint8)

            # Send to Arduino
            streamer.send_frame(frame)

            frame_count += 1

            # Print stats every second
            if frame_count % args.fps == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"Frames: {frame_count}, Actual FPS: {actual_fps:.1f}")

            # Frame timing
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        streamer.close()
        print("Done!")

if __name__ == '__main__':
    main()
