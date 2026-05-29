#!/usr/bin/env python3
"""Real-time gyro fleet visualizer — reads binary-framed v1 telemetry from the sniffer board.

The board legend is seeded from the fleet catalog (festicorn/catalog/boards.yaml):
every `role: bulb-sender` board is listed up front, keyed by the last two bytes of
its MAC. Each entry shows how stale its last packet is ("3s", "2m", "1h", "4d") or
"?" if we have never heard from it this session. Senders that aren't in the catalog
are added to the legend as they appear on the wire.
"""

import sys
import struct
import time
import collections
from pathlib import Path

import serial
import numpy as np
import yaml
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

SERIAL_PORT = "/dev/cu.usbmodem1121201"
BAUD = 460800
WINDOW_SEC = 10.0
COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (255, 180, 100), (180, 100, 255),
]
AMAG_FS = 57000.0
COUNTS_PER_G = 32768.0 / 4.0
COUNTS_PER_DPS = 32768.0 / 1000.0

V1_SIZE = 16
FRAME_HEADER = bytes([0xA5, 0x5A])

# Fleet catalog: festicorn/catalog/boards.yaml. From this file
# (festicorn/gyro-sense/tools/) the repo root is three parents up.
CATALOG_PATH = Path(__file__).resolve().parents[3] / "festicorn" / "catalog" / "boards.yaml"


def mac_short(mac_str):
    """Last two bytes of a colon MAC string, e.g. '14:63:93:70:0B:88' -> '0B:88'."""
    parts = mac_str.strip().split(":")
    return f"{parts[-2]}:{parts[-1]}".upper()


def load_catalog_senders():
    """Return ordered list of (mac_short) for every bulb-sender in the catalog.

    Missing/unparseable catalog is non-fatal — the legend just starts empty and
    fills in as senders appear on the wire.
    """
    try:
        data = yaml.safe_load(CATALOG_PATH.read_text())
    except (OSError, yaml.YAMLError) as e:
        print(f"Catalog not loaded ({e}); legend will populate from wire only.", file=sys.stderr)
        return []
    shorts = []
    for board in data.get("boards") or []:
        if board.get("role") == "bulb-sender" and board.get("mac"):
            short = mac_short(board["mac"])
            if short not in shorts:
                shorts.append(short)
    return shorts


def fmt_age(seconds):
    """Human staleness: '?' if never seen, else compact s/m/h/d."""
    if seconds is None:
        return "?"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def mag_from_byte(b):
    n = b / 255.0
    return n * n * AMAG_FS


def amag_g(b):
    return mag_from_byte(b) / COUNTS_PER_G


def gmag_dps(b):
    return mag_from_byte(b) / COUNTS_PER_DPS


def axis_mean_g(m):
    return (m * 256.0) / COUNTS_PER_G


def parse_v1(data):
    seq, ax_max, ay_max, az_max, ax_min, ay_min, az_min, \
        ax_mean, ay_mean, az_mean, amag_max, amag_mean, \
        gmag_max, gmag_mean, flags = struct.unpack("<HbbbbbbbbbBBBBB", data[:16])
    return {
        "seq": seq,
        "amag_max_g": amag_g(amag_max), "amag_mean_g": amag_g(amag_mean),
        "gmag_max_dps": gmag_dps(gmag_max), "gmag_mean_dps": gmag_dps(gmag_mean),
        "grav_x": axis_mean_g(ax_mean), "grav_y": axis_mean_g(ay_mean), "grav_z": axis_mean_g(az_mean),
        "clip": (flags >> 4) & 0x0F, "sat": flags & 0x0F,
    }


class SenderData:
    def __init__(self, mac_short, color):
        self.mac_short = mac_short
        self.color = color
        self.times = collections.deque()
        self.amag_max = collections.deque()
        self.gmag_max = collections.deque()
        self.last_seen = None  # monotonic-relative time of last packet; None = never

    def add(self, t, pkt):
        self.times.append(t)
        self.amag_max.append(pkt["amag_max_g"])
        self.gmag_max.append(pkt["gmag_max_dps"])
        self.last_seen = t

    def prune(self, cutoff):
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
            self.amag_max.popleft()
            self.gmag_max.popleft()


class SerialReader(QtCore.QThread):
    packet_received = QtCore.pyqtSignal(str, dict)  # mac_short, parsed_pkt

    def __init__(self, port, baud):
        super().__init__()
        self.port = port
        self.baud = baud
        self._running = True

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.05)
        except serial.SerialException as e:
            print(f"Serial error: {e}", file=sys.stderr)
            return

        buf = bytearray()
        while self._running:
            chunk = ser.read(256)
            if not chunk:
                continue
            buf.extend(chunk)

            while True:
                idx = buf.find(FRAME_HEADER)
                if idx < 0:
                    buf.clear()
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < 3:
                    break
                frame_len = buf[2]
                total = 3 + frame_len + 1  # header(2) + len(1) + payload(frame_len) + xor(1)
                if len(buf) < total:
                    break

                xor8 = 0
                for i in range(2, 3 + frame_len):
                    xor8 ^= buf[i]
                if xor8 != buf[3 + frame_len]:
                    buf = buf[2:]
                    continue

                mac = buf[3:9]
                payload = buf[9:9 + frame_len - 6]
                buf = buf[total:]

                if len(payload) == V1_SIZE:
                    short = f"{mac[4]:02X}:{mac[5]:02X}"
                    pkt = parse_v1(bytes(payload))
                    self.packet_received.emit(short, pkt)

        ser.close()

    def stop(self):
        self._running = False
        self.wait(2000)


class VisualizerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gyro Fleet Visualizer")
        self.resize(1200, 700)

        self.senders: dict[str, SenderData] = {}
        self.color_idx = 0
        self.t0 = time.monotonic()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.gw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.gw)

        self.plot_amag = self.gw.addPlot(row=0, col=0, title="Accel Magnitude (g)")
        self.plot_gmag = self.gw.addPlot(row=1, col=0, title="Gyro Magnitude (dps)")

        for p in [self.plot_amag, self.plot_gmag]:
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("bottom", "Time", "s")

        self.plot_amag.setYRange(0, 4)
        self.plot_gmag.setYRange(0, 500)

        self.curves = {}  # mac_short -> dict of curve objects

        # Per-plot legend: colored MAC + age text anchored to the top-right
        # corner. Repositioned each frame to track the scrolling x-range.
        # (plot, y_top) pairs — y_top is the plot's fixed YRange max.
        self.legend_targets = [(self.plot_amag, 4.0), (self.plot_gmag, 500.0)]
        self.legend_items = []
        for plot, _ in self.legend_targets:
            ti = pg.TextItem(anchor=(1, 0))
            plot.addItem(ti, ignoreBounds=True)
            self.legend_items.append(ti)

        # Seed the fleet from the catalog so the legend lists every known
        # bulb-sender immediately — unheard ones show "?" until a packet lands.
        for short in load_catalog_senders():
            self._get_sender(short)

        self.reader = SerialReader(SERIAL_PORT, BAUD)
        self.reader.packet_received.connect(self.on_packet)
        self.reader.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # 20 Hz refresh

    def _get_sender(self, mac_short):
        if mac_short not in self.senders:
            color = COLORS[self.color_idx % len(COLORS)]
            self.color_idx += 1
            self.senders[mac_short] = SenderData(mac_short, color)
        return self.senders[mac_short]

    def _ensure_curves(self, mac_short):
        """Lazily build plot curves once a sender actually has data."""
        if mac_short in self.curves:
            return
        color = self.senders[mac_short].color
        pen = pg.mkPen(color=color, width=2)
        # No in-plot legend — the color-matched bottom legend (MAC + age) is
        # authoritative and lists the whole fleet, including unheard boards.
        self.curves[mac_short] = {
            "amag": self.plot_amag.plot(pen=pen),
            "gmag": self.plot_gmag.plot(pen=pen),
        }

    def on_packet(self, mac_short, pkt):
        sd = self._get_sender(mac_short)
        self._ensure_curves(mac_short)
        t = time.monotonic() - self.t0
        sd.add(t, pkt)

    def update_plots(self):
        now = time.monotonic() - self.t0
        cutoff = now - WINDOW_SEC

        legend_parts = []
        for mac_short in sorted(self.senders):
            sd = self.senders[mac_short]
            sd.prune(cutoff)

            if mac_short in self.curves and len(sd.times) > 0:
                c = self.curves[mac_short]
                t = np.array(sd.times)
                c["amag"].setData(t, np.array(sd.amag_max))
                c["gmag"].setData(t, np.array(sd.gmag_max))

            age = None if sd.last_seen is None else now - sd.last_seen
            # Only annotate staleness once it's worth noting: "?" if never heard,
            # the age if more than a second behind, otherwise just the MAC.
            if age is None:
                suffix = "&nbsp;?"
            elif age > 1.0:
                suffix = f"&nbsp;{fmt_age(age)}"
            else:
                suffix = ""
            r, g, b = sd.color
            legend_parts.append(
                f'<span style="color:rgb({r},{g},{b})">{mac_short}{suffix}</span>'
            )

        for p, _ in self.legend_targets:
            p.setXRange(cutoff, now)

        html = (
            "<div style='text-align:right; font-family:Menlo,monospace; font-size:10pt'>"
            + "<br>".join(legend_parts) + "</div>"
            if legend_parts else
            "<div style='font-family:Menlo,monospace; font-size:10pt'>Waiting for packets…</div>"
        )
        for ti, (_, y_top) in zip(self.legend_items, self.legend_targets):
            ti.setHtml(html)
            ti.setPos(now, y_top)  # anchor (1,0) → top-right corner

    def closeEvent(self, event):
        self.reader.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, pg.mkColor(30, 30, 30))
    palette.setColor(palette.ColorRole.WindowText, pg.mkColor(200, 200, 200))
    app.setPalette(palette)
    pg.setConfigOption("background", (30, 30, 30))
    pg.setConfigOption("foreground", (200, 200, 200))

    win = VisualizerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
