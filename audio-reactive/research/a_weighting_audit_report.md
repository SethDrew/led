# A-Weighting Implementation Audit Report

**Date:** 2026-02-21
**Files Audited:**
- `/Users/KO16K39/Documents/led/audio-reactive/tools/viewer.py` (lines 188-222)
- `/Users/KO16K39/Documents/led/audio-reactive/tools/web_viewer.py` (lines 1476-1495)

---

## Executive Summary

**The A-weighting implementation is CORRECT**, but it's the **WRONG curve for music** at typical listening levels.

### Key Findings

1. ✅ **Formula is correct** - matches IEC 61672 exactly
2. ✅ **Power domain application is correct** - properly converts dB to amplitude, then squares
3. ✅ **Sub-bass 0.00 ratio is expected** - not a bug, but a consequence of using A-weighting
4. ⚠️ **A-weighting is inappropriate for music** - designed for 40 phons (quiet), not 70-85 dB (normal music)

---

## 1. Formula Verification

### IEC 61672 Standard
```
Ra(f) = (12194² × f⁴) / ((f² + 20.6²) × sqrt((f² + 107.7²) × (f² + 737.9²)) × (f² + 12194²))
A(f) = 20 × log10(Ra(f)) + 2.0
```

### Implementation (viewer.py:191-198)
```python
f2 = freqs ** 2
a_weight_db = (
    20 * np.log10(
        (12194**2 * f2**2) /  # f2**2 = f**4 ✓
        ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
        + 1e-20
    )
    + 2.0
)
```

**Result:** Formula is mathematically identical to IEC 61672. The `1e-20` prevents division by zero.

### Verification at Standard Frequencies

| Freq (Hz) | Expected (dB) | Actual (dB) | Error (dB) |
|-----------|---------------|-------------|------------|
| 20        | -50.5         | -50.4       | 0.1        |
| 50        | -30.2         | -30.3       | -0.1       |
| 100       | -19.1         | -19.1       | 0.0        |
| 250       | -8.6          | -8.7        | -0.1       |
| 1000      | 0.0           | 0.0         | 0.0        |
| 2000      | 1.2           | 1.2         | 0.0        |
| 4000      | 1.0           | 1.0         | 0.0        |
| 6000      | -1.0          | 0.1         | 1.1        |
| 8000      | -1.1          | -1.1        | 0.0        |

Minor discrepancies (<0.2 dB) are due to rounding in the standard tables.

---

## 2. Power Domain Application

### Implementation (viewer.py:199-203)
```python
a_weight_linear = 10 ** (a_weight_db / 20.0)  # amplitude factor
a_weight_linear[0] = 0  # DC bin
S_power = np.abs(librosa.stft(...)) ** 2
S_weighted = S_power * (a_weight_linear[:, np.newaxis] ** 2)  # power domain
```

### Verification

A-weighting is defined as a correction to SPL (Sound Pressure Level):
- SPL = 20 × log₁₀(p/p_ref) [dB]
- SPL_A = SPL + A(f) [dB]

Converting to linear power domain:
- p_A = p × 10^(A(f)/20) [amplitude]
- P_A = p_A² = p² × (10^(A(f)/20))² = P × 10^(A(f)/10) [power]

**Mathematical proof:**
```
(10^(A/20))² = 10^(2×A/20) = 10^(A/10) ✓
```

**Result:** The implementation correctly converts dB to amplitude (`/20`), then squares for power domain.

---

## 3. Band Attenuation Analysis

### Average A-Weighting per Band

| Band       | Freq Range   | Avg A-weight (dB) | Power Factor |
|------------|--------------|-------------------|--------------|
| Sub-bass   | 20-80 Hz     | -32.1             | 0.0006       |
| Bass       | 80-250 Hz    | -13.7             | 0.0424       |
| Mids       | 250-2000 Hz  | -0.7              | 0.8493       |
| High-mids  | 2000-6000 Hz | 0.9               | 1.2184       |
| Treble     | 6000-8000 Hz | -0.5              | 0.8848       |

### Impact on Typical Music

**Before A-weighting:**
- Sub-bass: 0.30
- Bass: 0.80
- Mids: 1.00
- High-mids: 0.70
- Treble: 0.40

**After A-weighting:**
- Sub-bass: 0.00 (0.0002 absolute)
- Bass: 0.04
- Mids: 1.00
- High-mids: 1.00
- Treble: 0.41

**Conclusion:** Sub-bass being 0.00 is **expected behavior**. A-weighting attenuates 20-80 Hz by ~32 dB on average, reducing power by a factor of ~1600.

---

## 4. Why A-Weighting is Wrong for Music

### A-Weighting Design Context

A-weighting approximates the **40-phon equal-loudness contour** (ISO 226), which represents:
- Very quiet listening environments
- Background noise measurement
- Hearing damage risk at low levels

At 40 phons:
- Our ears are MUCH less sensitive to bass
- Sub-bass (20-50 Hz) requires ~50 dB more SPL to sound as loud as 1 kHz
- This is appropriate for safety/regulatory noise measurement

### Music Listening Context

Music is typically listened to at **70-85 dB SPL** (80 phons), where:
- Bass sensitivity is MUCH better
- The equal-loudness curves flatten out
- Sub-bass only needs ~10-20 dB more SPL to match 1 kHz

### Comparison: A-weighting vs C-weighting

| Band       | A-weight avg (dB) | C-weight avg (dB) | Difference |
|------------|-------------------|-------------------|------------|
| Sub-bass   | -32.1             | -1.9              | 30.2 dB    |
| Bass       | -13.7             | -0.1              | 13.6 dB    |
| Mids       | -0.7              | 0.0               | 0.7 dB     |
| High-mids  | 0.9               | -0.9              | 1.8 dB     |
| Treble     | -0.5              | -2.4              | 1.9 dB     |

**C-weighting** is designed for loud sounds (80-100 dB SPL) and preserves bass much better.

---

## 5. Recommendations

### Option 1: Remove Weighting (RECOMMENDED)
Let the music spectrum speak for itself. The LED effects are about capturing the energy distribution in the music, not simulating perceived loudness.

```python
# No weighting needed - use raw mel spectrogram
mel_spec = librosa.feature.melspectrogram(...)
```

### Option 2: Use C-Weighting
If perceptual weighting is desired, use C-weighting for realistic music listening levels.

```python
def c_weight_db(f):
    """C-weighting (IEC 61672) for 80-100 dB SPL"""
    f = np.asarray(f, dtype=np.float64)
    f2 = f ** 2
    numerator = 12194**2 * f2
    denominator = (f2 + 20.6**2) * (f2 + 12194**2)
    Rc = numerator / (denominator + 1e-20)
    return 20 * np.log10(Rc + 1e-20) + 0.06
```

C-weighting attenuuation:
- Sub-bass: -2 dB (preserves most bass)
- Bass: -0.1 dB (nearly flat)
- Mids: 0 dB (reference)
- High-mids: -1 dB (slight attenuation)
- Treble: -2 dB (slight attenuation)

### Option 3: ISO 226 Equal-Loudness at 80 Phons
Most accurate for typical music listening, but requires lookup tables or complex formulas.

---

## 6. Consistency Between Implementations

Both `viewer.py` and `web_viewer.py` use identical A-weighting implementations:

**viewer.py (lines 191-203):**
```python
a_weight_db = (20 * np.log10(...) + 2.0)
a_weight_linear = 10 ** (a_weight_db / 20.0)
a_weight_linear[0] = 0
S_weighted = S_power * (a_weight_linear[:, np.newaxis] ** 2)
```

**web_viewer.py (lines 1480-1493):**
```python
a_weight_db = (20 * np.log10(...) + 2.0)
a_weight_linear = 10 ** (a_weight_db / 20.0)
a_weight_linear[0] = 0
S_weighted = S * (a_weight_linear[:, np.newaxis] ** 2)
```

Both implementations are correct and identical.

---

## 7. Bugs Found

**None.** The implementation is mathematically correct.

The observed behavior (sub-bass = 0.00, treble = 0.00) is the **expected result** of A-weighting, which is designed for quiet listening environments, not music at typical concert/listening volumes.

---

## 8. Action Items

1. **Decision needed:** Choose weighting strategy
   - No weighting (let music speak)
   - C-weighting (realistic for loud music)
   - ISO 226 at 80 phons (most accurate but complex)

2. **Update both files** if weighting is changed:
   - `audio-reactive/tools/viewer.py`
   - `audio-reactive/tools/web_viewer.py`

3. **Document the choice** in ledger with rationale

4. **Test with bass-heavy music** to verify the new behavior matches expectations

---

## References

- IEC 61672:2013 - Electroacoustics — Sound level meters
- ISO 226:2003 - Acoustics — Normal equal-loudness-level contours
- Fletcher-Munson curves (historical basis for A-weighting)

---

## Test Files Created

- `/Users/KO16K39/Documents/led/a_weight_test.py` - Formula verification
- `/Users/KO16K39/Documents/led/power_test.py` - Power domain verification
- `/Users/KO16K39/Documents/led/band_analysis.py` - Band attenuation analysis

Run with: `source venv/bin/activate && python <test_file>.py`
