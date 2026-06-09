# BS-26 Kay Lab 508-25 — Knob & Switch Hardware Reference

Golden copy. All calibration measurements taken at 70°F in a temp-controlled room.

## Ground Reference Environments

| Environment | Description |
|---|---|
| **USB-grounded** | Laptop USB connected — provides clean GND reference via cable shield |
| **Ungrounded** | Battery/wall power only — no laptop GND. ADC readings shift down due to floating ground plane |

---

## Controls

### DC Switch (GPIO 14)
- Type: Toggle, 2-state
- Sense: Voltage divider, HIGH=ON, LOW=OFF
- Debounce: 100ms
- No ground sensitivity (digital)

### AC Switch (GPIO 13)
- Type: Toggle, 2-state
- Sense: ~100Ω closed (oxidized contacts), open when ON (inverted logic)
- Debounce: 100ms
- Known issue: contact oxidation; may drift

### Hundreds Knob (GPIO 35, ESP32 ADC)
- Type: 5-position rotary switch
- Pullup: 10kΩ to 3.3V
- Resistance ladder: 8.8kΩ / 6.6kΩ / 4.4kΩ / 2.2kΩ / short
- Hysteresis: ±40 counts

**Threshold ladder (descending):**
```
HUNDREDS_THRESH[4] = {1630, 1290, 835, 285}
```

**Measured ADC by position:**

| Pos | Resistance | USB GND | No GND | Delta |
|-----|-----------|---------|--------|-------|
| 1 | 8.8kΩ | 1780 | 1731 | -49 |
| 2 | 6.6kΩ | 1480 | 1437 | -43 |
| 3 | 4.4kΩ | 1100 | 1052 | -48 |
| 4 | 2.2kΩ | 570 | 526 | -44 |
| 5 | short | 0 | 0 | 0 |

USB values from manual measurement (±20 counts). No-GND medians from ESP-NOW capture (stdev 8–17).
All positions decode correctly in both modes — gaps are 300+ counts, shifts are ~45.

### Tens Knob (GPIO 32, ESP32 ADC)
- Type: 10-position potentiometer
- Resistance: 10MΩ full range
- Pullup: 5.6MΩ to 3.3V
- Cap filtering: yes (to GND)
- Hysteresis: ±40 counts

**Threshold ladder (descending, unified for both ground modes):**
```
TENS_THRESH[9] = {3133, 2095, 1942, 1767, 1565, 1304, 995, 602, 197}
```

**Measured ADC by position:**

| Pos | USB GND | No GND | Delta | Threshold below | Margin to threshold |
|-----|---------|--------|-------|-----------------|---------------------|
| 0 | 4095 | 4095 | 0 | — | — |
| 1 | 2171 | 2160 | -11 | 2095 | 65 |
| 2 | 2031 | 2023 | -8 | 1942 | 81 |
| 3 | 1861 | 1854 | -7 | 1767 | 87 |
| 4 | 1681 | 1674 | -7 | 1565 | 109 |
| 5 | 1456 | 1439 | -17 | 1304 | 135 |
| 6 | 1167 | 1169 | +2 | 995 | 172 |
| 7 | 824 | 808 | -16 | 602 | 206 |
| 8 | 396 | 394 | -2 | 197 | 197 |
| 9 | 0 | 0 | 0 | — | — |

USB medians from settled per-position capture. No-GND medians from ESP-NOW settled captures (stdev 3–14).
Tightest boundary: 1|2 at half-margin 64 counts. All positions decode in both modes.

### Ones Knob (GPIO 34, ESP32 ADC)
- Type: 11-position potentiometer (labeled 1.02, 2–11)
- Resistance: 1MΩ full range
- Pullup: 1MΩ to 3.3V
- No cap filtering
- Drives effect select — chatter here is worst for UX
- Hysteresis: ±15 counts (reduced from 40 — tighter spacing requires lower hysteresis)

**Threshold ladder (descending, unified for both ground modes):**
```
ONES_THRESH[11] = {1950, 1846, 1736, 1624, 1478, 1330, 1161, 971, 743, 408, 93}
```

**Measured ADC by position:**

| Pos | Label | USB GND | No GND | Delta | Threshold below | Margin to threshold |
|-----|-------|---------|--------|-------|-----------------|---------------------|
| 1 | 1.02 | 1871 | 1917 | +46 | 1846 | 25 |
| 2 | 2 | 1767 | 1822 | +55 | 1736 | 31 |
| 3 | 3 | 1663 | 1705 | +42 | 1624 | 39 |
| 4 | 4 | 1529 | 1585 | +56 | 1478 | 51 |
| 5 | 5 | 1391 | 1427 | +36 | 1330 | 61 |
| 6 | 6 | 1229 | 1270 | +41 | 1161 | 68 |
| 7 | 7 | 1057 | 1093 | +36 | 971 | 86 |
| 8 | 8 | 846 | 885 | +39 | 743 | 103 |
| 9 | 9 | 614 | 640 | +26 | 408 | 206 |
| 10 | 10 | 187 | 202 | +15 | 93 | 94 |
| 11 | 11 | 0 | 0 | 0 | — | — |

USB medians from `bs26/retrofit/tools/knobA_data.csv`. No-GND medians from ESP-NOW settled captures (stdev 3–9).
Ones shifts UP without USB (+2.5–4.6%), opposite to tens/hundreds. Tightest boundary: 1|2 at half-margin 24 counts.
All positions decode in both modes with HYST=15.

### Decade Knob (ADS1115 on I2C, SDA=23 SCL=22, addr 0x48)

Three taps on a single decade potentiometer, read via resistor dividers.

**ADS1115 config:**
- Default gain: `GAIN_ONE` (±4.096V, 0.125 mV/bit)
- For inner (ones) tap: `GAIN_TWO` (±2.048V, 0.0625 mV/bit)
- VCC reference: 3.394V measured

#### Decade Outer (hundreds, ADS Ch0)
- Range: 0–1200Ω
- Pullup: 1kΩ to 3.3V
- Cap: 1µF to GND
- 12 positions

**LUT and thresholds:**
```
hundredsLUT[13] = {6, 2440, 4456, 6171, 7631, 8903, 10009, 10991, 11857, 12637, 13334, 13968, 14546}
hundredsThresh[11] = {2326, 4366, 6090, 7565, 8842, 9958, 10943, 11817, 12599, 13302, 13937}
```

#### Decade Mid (tens, interpolated from ADS Ch0)
- Derived from bracket interpolation between hundredsLUT entries
- `countsPerTen = (bracketHigh - bracketLow) / 10.0`
- 2-confirm debounce

**Ground shift observed:**

| Condition | ADS raw (pos 9/9) | Decoded mid |
|---|---|---|
| USB-grounded | ~13255 | 9 |
| Ungrounded | ~12778 | **2** (massive shift) |

Offset: -477 ADS counts. Bracket interpolation is very sensitive — a shift in the raw value moves the interpolated mid digit dramatically.

#### Decade Inner (ones tap, ADS Ch1)
- Range: 0–8.9Ω
- Pullup: 510Ω to 3.3V
- Cap: 1µF to GND
- 4-sample averaging before decode
- Resistance calc: `R = V × 510 / (3.394 - V)` where V = raw × 0.0000625
- 2-confirm debounce
- Ultra-low noise (±4 counts typical with USB GND)

**Ground shift observed:**

| Condition | ADS raw (pos 0) |
|---|---|
| USB-grounded | ~12 |
| Ungrounded | ~12 (stable) |

Inner tap unaffected — 510Ω pullup to low-R tap is ground-insensitive.

---

## Noise Rejection Stack

| Layer | Mechanism | Parameter |
|---|---|---|
| 1 | Hysteresis | ±40 counts (hundreds, tens), ±15 counts (ones) |
| 2 | Sample confirm | 3 consecutive (native), 2 consecutive (ADS) |
| 3 | ADS averaging | 4-sample mean for inner tap |
| 4 | Cap filtering | 1µF on decade taps, cap on tens GPIO |
| 5 | ADS ratiometric | 7.5k/7.5k VCC ref on A2, corrects decade for rail droop |

---

## Meter Outputs (DAC)

| Meter | GPIO | DAC max | Full-scale | Series R | Coil R | Auto-map |
|---|---|---|---|---|---|---|
| µA (brightness) | 25 (DAC1) | 230 | 500µA | 5.6kΩ | — | tens 0–9 → 0–90 on face |
| V (speed) | 26 (DAC2) | 245 | 500V | 5.6kΩ | 171.9Ω | hundreds 0–4 → 0–400 on face |

**Critical:** Use `digitalWrite(pin, LOW)` for true 0V. `dacWrite(pin, 0)` leaks ~80mV.

---

## Ground Shift Summary (2026-06-08)

All measurements via ESP-NOW sniffer (no USB to sender). 70°F temp-controlled room.

### Before correction

Knobs at: hundreds=2, tens=2, ones=1.02, decade=990. No reference divider or scaling.

| Reading | USB GND | No GND | Delta | Impact |
|---|---|---|---|---|
| hundreds raw | 1085 | 848 | **-237** | Could cross bin boundary |
| tens raw | 2035 | 1945 | **-90** | **Crossed threshold** (pos 2→3) |
| ones raw | 1900 | 1812 | **-88** | Held in bin (hysteresis saved it) |
| adsTens | 13255 | 12778 | **-477** | **decMid shifted 9→2** |
| adsOnes | 12 | 12 | 0 | Low-Z tap immune |

### After correction

Added 7.5k/7.5k voltage divider on ADS1115 A2 as a VCC reference channel. ADS readings scaled ratiometrically against this reference (baseline captured with USB GND). Same knob positions.

| Reading | USB GND | No GND | Delta | % | Impact |
|---|---|---|---|---|---|
| adsRef (7.5k div) | 13246 | 13214 | -32 | -0.24% | Reference channel tracks rail |
| adsTens (corrected) | 13251 | 13220 | -31 | -0.23% | **decMid holds correct position** |
| adsOnes | 12 | 12 | 0 | — | Low-Z tap immune (unchanged) |
| hundreds raw | 1075 | 1025 | -50 | -4.7% | Held in bin (hysteresis) |
| tens raw | 2030 | 2028 | -2 | -0.1% | Stable |
| ones raw | 1895 | 1885 | -10 | -0.5% | Stable |

### Analysis

The ADS1115 shift is pure VCC rail droop (~8mV, 0.24%) — eliminated by ratiometric scaling. The native ESP32 ADC shift on hundreds (4.7% vs 0.24% rail droop) is dominated by switched-cap input loading on the high-Z divider, not rail change. Lower-Z native channels (tens, ones) show minimal residual shift.

### 5-minute stability test (no USB, post-correction)

280 samples over 300 seconds. No thermal drift, no position changes.

| Channel | Spread | Stdev | Notes |
|---|---|---|---|
| rawH (10kΩ pullup) | 127 | 12.1 | Random jitter; WiFi TX causes occasional -80 count sag |
| rawT (5.6MΩ pullup) | 105 | 10.3 | Random jitter, no drift |
| rawO (1MΩ pullup) | 190 | 16.1 | Noisiest native channel, no drift |
| adsRef | 6 | 0.9 | Rock solid |
| adsTens | 6 | — | Rock solid |
| adsOnes | 4 | — | Rock solid |

### Decade outer full sweep (no USB, post-correction)

Swept positions 0→11→0 over 90 seconds. All positions decoded correctly.

| decOuter pos | adsTens range | Spread |
|---|---|---|
| 0 | 2209–2212 | 3 |
| 1 | 4269–4270 | 1 |
| 2 | 6000–6001 | 1 |
| 3 | 7488–7490 | 2 |
| 4 | 8771–8773 | 2 |
| 5 | 9894–9898 | 4 |
| 6 | 10880–10885 | 5 |
| 7 | 11763–11764 | 1 |
| 8 | 12544–12548 | 4 |
| 9 | 13250–13256 | 6 |
| 10 | 13889 | 0 |
| 11 | 14469–14473 | 4 |

---

## Calibration Data Files

Located in `/Users/sethdrew/Documents/projects/bs26/retrofit/tools/`:
- `hundreds_cal.csv` — rotary switch full calibration
- `hundreds_data.csv` — ADS1115 detailed measurements
- `tens_data.csv` — tens knob full sweep
- `ones_data.csv` — ones knob full sweep (ultra-stable)
- `knobA_data.csv` — native ESP32 ADC validation

## Source Code

- Current firmware: `festicorn/kaylab-knobs/src/main.cpp`
- Original retrofit: `~/Documents/projects/bs26/retrofit/src/main.cpp`
- Calibration scripts: `~/Documents/projects/bs26/retrofit/tools/`
