#!/usr/bin/env python3
"""Headless self-test for mic_bridge.

Verifies port discovery, mic tick, recording, and BACK-TO-BACK playback
(the real-world failure path that previously SIGTRAPped on macOS).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mic_bridge import Recorder, discover_bridge_port, RMS_SCALE_DEFAULT

bridge_port = discover_bridge_port()
print(f"bridge={bridge_port or '(none — mic-only)'}")
r = Recorder(bridge_port, RMS_SCALE_DEFAULT)
r.open()
time.sleep(0.5)
print(r.start())
time.sleep(3.0)
print(r.stop())
print("--- playback #1 ---")
print(r.playback())
print("--- playback #2 (regression check) ---")
print(r.playback())
print("--- live again? rms=", r.last_rms_u16)
time.sleep(0.5)
print("rms after restart:", r.last_rms_u16)
r.close()
print("OK")
