"""Stream Mac audio (via BlackHole) to ESP32 over serial — raw PCM."""
import pyaudio
import serial
import sys
import signal
import time
import struct
import array

SERIAL_PORT = "/dev/cu.usbserial-0001"
SERIAL_BAUD = 921600
SAMPLE_RATE = 22050
CHUNK = 256
GAIN = 16  # Volume multiplier (1=original, 2=2x, 4=4x, etc.)

def find_blackhole(pa):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "blackhole" in info["name"].lower() and info["maxInputChannels"] > 0:
            print(f"Found: {info['name']} (index {i})")
            return i
    return None

def main():
    pa = pyaudio.PyAudio()

    dev_idx = find_blackhole(pa)
    if dev_idx is None:
        print("BlackHole not found!")
        pa.terminate()
        sys.exit(1)

    # Open serial without triggering ESP32 reset
    print(f"Opening serial: {SERIAL_PORT} @ {SERIAL_BAUD}")
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=SERIAL_BAUD,
        timeout=0.1,
        dsrdtr=False,
        rtscts=False,
    )

    # Wait for ESP32 to be ready
    print("Waiting for ESP32 READY...")
    deadline = time.time() + 15
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(f"  [ESP32] {line}")
            if "READY" in line:
                break
        time.sleep(0.1)

    print(f"\nStreaming: {SAMPLE_RATE}Hz mono 16-bit -> serial")
    print("Press Ctrl+C to stop\n")

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=dev_idx,
        frames_per_buffer=CHUNK,
    )

    def cleanup(*args):
        print("\nStopping...")
        stream.stop_stream()
        stream.close()
        ser.close()
        pa.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    frames = 0
    while True:
        pcm = stream.read(CHUNK, exception_on_overflow=False)
        # Amplify
        samples = array.array('h', pcm)  # signed 16-bit
        for j in range(len(samples)):
            s = samples[j] * GAIN
            samples[j] = max(-32768, min(32767, s))  # clamp
        ser.write(samples.tobytes())
        frames += 1
        if frames % (SAMPLE_RATE // CHUNK) == 0:
            secs = frames * CHUNK / SAMPLE_RATE
            sys.stdout.write(f"\r  {secs:.0f}s streamed")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
