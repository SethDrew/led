"""PlatformIO pre-script: injects per-board facts from festicorn/catalog/boards.yaml
so pin mappings can't be hand-typed wrong.

Usage in platformio.ini:
    [env:canbus-eth1]
    extra_scripts = pre:../catalog/pio_board_lookup.py
    custom_board_id = led-eth-1

Upload ports are NOT resolved here — identify the board by MAC scan and pass
--upload-port explicitly (see boards.yaml header).
"""

Import("env")

import pathlib
import sys

import yaml

CATALOG = pathlib.Path(env["PROJECT_DIR"]).resolve().parent / "catalog" / "boards.yaml"


def fail(msg):
    sys.stderr.write(f"\n[catalog] ERROR: {msg}\n\n")
    env.Exit(1)


board_id = env.GetProjectOption("custom_board_id", "")
if not board_id:
    fail("this env uses pio_board_lookup.py but sets no custom_board_id")

catalog = yaml.safe_load(CATALOG.read_text())
boards = {b["id"]: b for b in catalog.get("boards", [])}
if board_id not in boards:
    fail(f"'{board_id}' not in {CATALOG} (known: {', '.join(sorted(boards))})")
board = boards[board_id]

can = board.get("can")
if can:
    env.Append(CPPDEFINES=[("CAN_TX_PIN", can["tx"]), ("CAN_RX_PIN", can["rx"])])
    print(f"[catalog] {board_id}: CAN_TX_PIN={can['tx']} CAN_RX_PIN={can['rx']} (from boards.yaml)")
