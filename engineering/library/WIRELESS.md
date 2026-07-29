# Wireless (ESP-NOW) — Synthesis & Recommendations

Consolidates the ESP-NOW findings from the rf-bench campaign (2026-07-27,
`festicorn/rf-bench/`) and the axismundi field measurements (2026-07-28).
Raw findings and full caveats live in the engineering ledger — entry IDs are
cited throughout; this document is the synthesis and the recommendations.
Per-board facts (MACs, defects, power pins) live in `festicorn/catalog/boards.yaml`.

## TL;DR

| Situation | Recommendation |
|---|---|
| Sender that must be reliable | Classic ESP32, or any board *screened* (see below) |
| C3 Super Mini as sender | Budget it as a weak transmitter (≥6.75 dB down as deployed); screen the unit; pin TX power per boards.yaml |
| C3 Super Mini as receiver | Fine — RX is healthy even on TX-defective units |
| Broadcast delivery expectations | Raw frame loss 25–85% is normal at range; design for it |
| Measuring delivery | Receiver-side seq gaps ONLY; never the send callback |
| On-change-once state packets | Fragile under broadcast — add send redundancy |

## The two delivery regimes (why "rock solid" and "80% loss" are both true)

- **Unicast** to a registered peer gets MAC-layer ACKs and automatic retries.
  The rf-bench pairwise matrix was 40/40 = 100% on every link, every family,
  every sub-cliff power level. Retries hide raw loss.
- **Broadcast** is one transmission, no ACK, no retry — and the send callback
  reports SUCCESS when the frame *leaves the antenna*, regardless of whether
  anyone heard it (ledger: esp32-c3-super-mini-tx-defect). The same bench
  measured 26–35% raw beacon loss at a remote position
  (esp-now-c3-supermini-tx-deficit-substitution).

All festicorn console links are broadcast. Judge them by the broadcast numbers.

## Is it "broadcast works on classic but not C3"?

No — the protocol behaves identically on both families. What differs is link
margin, and it stacks against the C3 Super Mini:

1. **Radiated power**: a non-defective C3 Super Mini was heard ≥6.75 dB weaker
   than a classic ESP32 at identical commanded power, same position, same
   receiver ("board as deployed" — antenna + layout, not silicon;
   esp-now-c3-supermini-tx-deficit-substitution). 6.75 dB off the link budget
   converts directly into raw broadcast loss at range.
2. **Per-unit defects**: two C3 Super Minis (different batches) brown out at
   exactly 16.00 dBm — 100% delivery at 15.75, 0% at 16.00, a 0.25 dB cliff
   (esp32-c3-super-mini-tx-defect). RX stays healthy on defective units.
3. **The cliff is duty-cycle dependent**: it is a regulator-brownout mechanism
   driven by sustained transmission, and it bites within seconds — 98% loss at
   50 Hz vs 0% at 1 Hz at the same commanded power. It also has a hard-reboot
   variant (rail sags far enough to reset the chip), and it is intermittent:
   a clean pass at a given power carries no safety information
   (esp-now-c3-brownout-cliff-duty-cycle-dependent).

A classic ESP32 broadcasting the same packets simply arrives with ~7+ dB more
margin, which at these distances is the difference between ~0% and ~50% loss.

Field measurement (axismundi, 2026-07-28, defective C3 kettle console pinned
at 15.75 dBm, receiver seq-gap accounting): raw broadcast loss **57%** and
**80%** in two runs while the console was being spun, then **0.02%**
after a reflash/reboot. RESOLVED by controlled A/B same day: the variable was
TX duty cycle, not uptime or reboot — the "clean" run was idle at heartbeat
rate. At sustained 50 Hz the same unit lost 97-99% at 15.75 dBm and 0.00% at
14.75 dBm across four alternating blocks each; it is now pinned at 12.75 dBm
(~41k-packet validation, ~0.01% loss)
(esp-now-c3-brownout-cliff-duty-cycle-dependent).

## Experiments run (methods, so they can be rerun)

| Experiment | Method | Where |
|---|---|---|
| Pairwise delivery matrix | 12 links × unicast bursts, ACK-counted, TX sweep | rf-bench `tools/rfbench.py`, `data/matrix_*.json` |
| TX-power cliff | Unicast ACK vs commanded power, 3 independent promiscuous receivers | `data/cliff_*.json`; ledger esp32-c3-super-mini-tx-defect |
| Commandable TX range | `esp_wifi_set_max_tx_power` readback sweep | ledger esp-now-tx-power-ceiling-c3-vs-classic |
| Transaction overhead | ACK-latency floor vs payload (16/64/250 B) | ledger esp-now-classic-esp32-fixed-213us-transaction-overhead |
| TX/RX balance | 4-board RSSI reciprocity matrix | ledger esp-now-rssi-reciprocity-method-and-limits |
| Radiated-power substitution | Same position, same commanded power, same fixed receiver; broadcast beacon seq-gap loss + unicast ACK | `tools/node_test.py`; ledger esp-now-c3-supermini-tx-deficit-substitution |
| Field broadcast loss | Receiver ring-buffers every arrival (ms, seq, enc), drained to serial; seq gaps = exact loss | axismundi.cpp RF diagnostics (2026-07-28) |

## Dev workflow

- **Screening a new board**: sweep TX power to ≥16 dBm under *sustained
  broadcast at production rate* (50 Hz for ≥60 s per level), loss counted by an
  independent receiver. Sparse probes and unicast ACKs are structurally blind
  to the brownout failure — the kettle unit passed 90/90 sparse unicast at low
  power, and later passed clean 30 s blocks at a power that also caused 97-99%
  loss and a chip reset in other blocks. One clean pass proves nothing.
- **Measuring a link**: put a seq counter in every packet; count gaps on the
  receiver. Sender-side callbacks are meaningless for broadcast. Watch for
  duplicate deliveries (seq delta 0) — count them separately from gaps.
- **Comparing chips at matched power**: 27 quarter-dBm (6.75 dBm) is the only
  level both families accept exactly (esp-now-tx-power-ceiling-c3-vs-classic).
  Set power after the radio is fully configured, then read it back.
- **Congestion trap**: N simultaneous broadcasters produce collision loss that
  reads as "marginal TX" on an innocent board. Diagnose boards in isolated
  pairs.
- **Serial ports**: identify boards by MAC, never by port name; usbserial
  (CH340) boards reset on ANY open under macOS, and multiple concurrent opens
  of the same port silently interleave garbage.

## Production workflow

- **Role assignment**: put classic ESP32 (or screened, externally-antenna'd
  boards) on links that must not degrade; C3 Super Minis are fine as
  receivers and acceptable as senders only with redundancy and screening.
- **Power pinning**: per-unit TX limits live in boards.yaml and in each
  sender's firmware (e.g. kettle-knob pins 12.75 dBm). Pin ≥2 dB below the
  *load-measured* cliff (sustained-rate, not sparse-probe). On a
  brownout-limited board the failure surface is non-monotonic in commanded
  power: never raise power in response to loss — extra commanded power buys
  no radiated power and collapses the PA.
- **State packets**: absolute, idempotent state + seq in every packet, so
  arbitrary loss and duplication are survivable and measurable.
- **Send redundancy**: on-change-once does not survive broadcast loss. Options,
  in increasing airtime: (a) repeat each changed state 3× over ~40 ms,
  plus 1 Hz heartbeat; (b) stream continuously at 20–50 Hz while active.
  Airtime cost is small either way (a 13 B frame is ~1.1 ms; 50 Hz ≈ 5%
  duty), but (a) is the polite default on a shared channel.
- **Rates**: ESP-NOW does not rate-adapt (1 Mbps, ~8 µs/byte + ~1 ms fixed);
  practical ceiling ~750–900 txn/s. Irrelevant at console rates (≤60 Hz).

## Open questions

- Receiver Wi-Fi power save: one A/B (WIFI_PS_NONE on the axismundi receiver)
  showed no improvement; single trial against a time-varying link — weak
  evidence, worth rerunning on a stable link.
- Multi-console + duck congestion behavior on channel 1 at full deployment.
