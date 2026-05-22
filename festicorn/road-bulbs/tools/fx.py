#!/usr/bin/env python3
"""Send effect commands to road-bulbs ESP32 via UDP port 4211.

Usage:
    fx.py gravity      (or g)
    fx.py sparkle      (or s)
    fx.py simple       (or S)
    fx.py fire         (or m)
    fx.py flicker      (or f)
    fx.py bloom        (or b)
    fx.py              (interactive mode — press key to switch)
"""
import socket
import sys

ESP_IP = "192.168.86.127"
CMD_PORT = 4211

EFFECTS = {
    "gravity": "g", "g": "g",
    "sparkle": "s", "s": "s",
    "simple": "S", "S": "S",
    "fire": "m", "m": "m",
    "flicker": "f", "f": "f",
    "bloom": "b", "b": "b",
}

def send(cmd: str, ip: str = ESP_IP):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(cmd.encode(), (ip, CMD_PORT))
    sock.close()
    print(f"→ {ip}:{CMD_PORT} '{cmd}'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        ip = sys.argv[2] if len(sys.argv) > 2 else ESP_IP
        cmd = EFFECTS.get(name)
        if not cmd:
            print(f"Unknown effect: {name}")
            print(f"Options: {', '.join(k for k in EFFECTS if len(k) > 1)}")
            sys.exit(1)
        send(cmd, ip)
    else:
        print("Interactive mode. Keys: g=gravity s=sparkle S=simple m=fire f=flicker b=bloom  q=quit")
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                c = sys.stdin.read(1)
                if c in ('q', '\x03'):
                    break
                if c in EFFECTS:
                    send(EFFECTS[c])
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print()
