import serial
import time

PORT = "/dev/cu.usbserial-11430"
BAUD = 460800
DURATION = 3.0

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"FAIL open {PORT}: {e}")
        return

    buf = bytearray()
    values = []
    start = time.time()
    while time.time() - start < DURATION:
        chunk = ser.read(256)
        if chunk:
            buf.extend(chunk)
        i = 0
        while i <= len(buf) - 3:
            if buf[i] == 0xFC:
                hi = buf[i + 1]
                lo = buf[i + 2]
                v = (hi << 8) | lo
                if 0 <= v <= 1023:
                    values.append(v)
                    i += 3
                    continue
            i += 1
        del buf[:i]
    ser.close()

    print(f"frames found: {len(values)}")
    if values:
        print(f"min: {min(values)}  max: {max(values)}  range: {max(values)-min(values)}")
        step = max(1, len(values) // 20)
        print("samples:", values[::step][:20])
    else:
        print("no 0xFC frames seen")

if __name__ == "__main__":
    main()
