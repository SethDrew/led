// Duck console features + the energy waterfall, hardware-independent.
//
// Shared by festicorn/axismundi/src/axismundi.cpp (the installation) and
// festicorn/axismundi/viz/src/axismundi_fx.h (the 3D visualizer), so the two
// cannot drift — same contract as lib/kettle_energy.
//
// Two stages, both consuming DuckPacketV1:
//
//   1. Features. Mic RMS → energy + onset through an adaptive floor and a
//      leaky ceiling. Gravity vector → hue through the original-duck sparkle
//      interaction, unchanged in shape: angle away from a calibrated rest
//      vector sweeps the shared hue arc over 0..180°, and the blend onto the base
//      water color ramps in over the first 40° past a 10° deadzone. Rest is
//      whatever the duck was doing for its first two seconds of contact, so
//      "upright" is wherever it happens to be standing.
//
//   2. Waterfall. Energy is poured in at the head of every strip and falls
//      along it, decaying as it goes. Each cell carries the hue and blend it
//      was poured with, so turning the duck recolors the water at the source
//      and the new color runs down the strip — the rotation reads as a moving
//      front, not a global recolor.
//
// Both hosts compose in the same 0..255 linear space and apply their own
// brightness/display transform afterwards, which is what lets the waterfall
// write straight into the caller's composed frame.

#ifndef DUCK_ENERGY_H
#define DUCK_ENERGY_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <duck_packet_v1.h>
#include <oklch_lut.h>
#include <hue_arc.h>

#define DUCK_TIMEOUT_MS       3000
#define DUCK_RMS_FLOOR_MIN   20000.0f
#define DUCK_RMS_CEIL_MIN   130000.0f
#define DUCK_ONSET_FLOOR      6000.0f
// Noise gate. Both thresholds are ratios against the adaptive floor, so the
// gate tracks the room instead of pinning an absolute level: it opens at
// +10.9 dB over ambient and closes at +6.8 dB, and the dB scale below starts
// at the close ratio so energy passes through zero exactly as the gate shuts.
#define DUCK_FLOOR_HEADROOM      2.2f
#define DUCK_GATE_OPEN_RATIO     3.5f
#define DUCK_GATE_HOLD_S         0.25f
#define DUCK_FLOOR_LEAK          0.005f
#define DUCK_FLOOR_SNAP_EPS      0.05f
#define DUCK_FLOOR_SOFT_SIG      0.6f
#define DUCK_FLOOR_TAU           0.11f
#define DUCK_CEIL_LEAK_K         0.0025f
#define DUCK_ONSET_REFRAC        0.120f
#define DUCK_ONSET_RISE_FRAC     0.5f
#define DUCK_ONSET_FAST_TAU      0.030f
#define DUCK_ONSET_SLOW_TAU      0.300f
#define DUCK_TILT_DEAD_DEG      10.0f
#define DUCK_HUE_MAX_DEG       180.0f
#define DUCK_BLEND_RANGE_DEG    40.0f
#define DUCK_CAL_MS           2000

#define DUCK_WF_SPEED_NORM  0.30f   // strip fractions per second
#define DUCK_WF_DECAY_K     0.45f
#define DUCK_WF_IDLE_LEVEL  0.0f
#define DUCK_WF_ONSET_GAIN  0.9f
#define DUCK_WF_GRAIN       0.45f   // birth-level jitter, 0 = smooth sheet
#define DUCK_WF_DEADBAND    0.02f
#define DUCK_WF_MAX_LEN     512     // longest strip either host renders (297)

struct DuckFeatures {
    float energy, onset, hueIdx, tilt;

    float floorV, ceilV;
    int   belowCnt;
    float onsetFast, onsetSlow, onsetLock;
    float gateHold;
    bool  gateOpen;

    float hueRotDeg;
    float restAx, restAy, restAz;
    float calSumAx, calSumAy, calSumAz;
    uint32_t calSamples, calStartMs;
    uint32_t lastUpMs;
    uint8_t  calSeq;
    bool     calibrated;
};

static inline float duckClampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// The OKLCH LUT is indexed by OKLCH hue, which runs 15-55 deg off RGB hue and
// not by a constant, so indexing it with an arc degree would put the duck's arc
// somewhere other than every other layer's. Invert the LUT once into RGB-hue
// buckets so the duck lands on the hue the arc actually asked for.
static uint8_t duckOklchIdx(float hsvDeg) {
    static uint8_t rev[256];
    static bool built = false;
    if (!built) {
        int16_t tmp[256];
        for (int i = 0; i < 256; i++) tmp[i] = -1;
        for (int i = 0; i < 256; i++) {
            float r = oklchConstL[i][0], g = oklchConstL[i][1], b = oklchConstL[i][2];
            float mx = fmaxf(r, fmaxf(g, b)), mn = fminf(r, fminf(g, b));
            float d = mx - mn;
            if (d <= 0.0f) continue;
            float h;
            if (mx == r)      h = 60.0f * fmodf((g - b) / d + 6.0f, 6.0f);
            else if (mx == g) h = 60.0f * ((b - r) / d + 2.0f);
            else              h = 60.0f * ((r - g) / d + 4.0f);
            tmp[(int)(hueWrap360(h) * (256.0f / 360.0f)) & 0xFF] = (int16_t)i;
        }
        int last = -1;
        for (int i = 511; i >= 0; i--) if (tmp[i & 0xFF] >= 0) { last = tmp[i & 0xFF]; break; }
        for (int i = 0; i < 256; i++) {
            if (tmp[i] >= 0) last = tmp[i];
            rev[i] = (uint8_t)(last < 0 ? 0 : last);
        }
        built = true;
    }
    return rev[(int)(hueWrap360(hsvDeg) * (256.0f / 360.0f)) & 0xFF];
}

static inline float duckLerpf(float a, float b, float t) { return a + (b - a) * t; }

static inline void duckResetAudio(DuckFeatures &f) {
    f.energy = 0.0f;
    f.onset = 0.0f;
    f.floorV = 0.0f;
    f.ceilV = DUCK_RMS_CEIL_MIN;
    f.belowCnt = 0;
    f.onsetFast = 0.0f;
    f.onsetSlow = 0.0f;
    f.onsetLock = 0.0f;
    f.gateHold = 0.0f;
    f.gateOpen = false;
}

static inline void duckResetTilt(DuckFeatures &f) {
    f.calibrated = false;
    f.calSumAx = f.calSumAy = f.calSumAz = 0.0f;
    f.calSamples = 0;
    f.calStartMs = 0;
    f.calSeq = 0;
    f.tilt = 0.0f;
}

static inline void duckFeaturesReset(DuckFeatures &f) {
    duckResetAudio(f);
    duckResetTilt(f);
    f.hueIdx = 0.0f;
    f.hueRotDeg = 0.0f;
    f.restAx = 0.0f; f.restAy = 0.0f; f.restAz = 1.0f;
    f.lastUpMs = 0;
}

static inline float duckDecodeRms(uint8_t b) {
    float r = (float)b / 255.0f;
    return r * r * DUCK_RMS_FS;
}

static inline void duckUpdateFloor(DuckFeatures &f, float rms, float dt) {
    if (f.floorV < 1.0f) {
        f.floorV = fmaxf(rms, DUCK_RMS_FLOOR_MIN);
        return;
    }
    if (rms < f.floorV * (1.0f + DUCK_FLOOR_SNAP_EPS)) {
        if (++f.belowCnt >= 3) {
            float target = fmaxf(rms, DUCK_RMS_FLOOR_MIN);
            f.floorV += fminf(1.0f, dt / DUCK_FLOOR_TAU) * (target - f.floorV);
        }
    } else {
        f.belowCnt = 0;
        float d = (rms / fmaxf(f.floorV, 1.0f) - 1.0f) / DUCK_FLOOR_SOFT_SIG;
        f.floorV *= (1.0f + DUCK_FLOOR_LEAK * dt * expf(-(d * d)));
    }
    f.floorV = fmaxf(f.floorV, DUCK_RMS_FLOOR_MIN);
}

// live = a fresh packet inside DUCK_TIMEOUT_MS. nowMs only needs to be a
// monotonic host clock; it is compared against itself for the calibration
// window, never against the packet's own uptime. hueRotDeg is the shared
// rotation base, which the duck adds to its own full-wheel hue so its water
// turns with the rest of the palette.
static inline void duckFeaturesUpdate(DuckFeatures &f, const DuckPacketV1 &pkt,
                                      float dt, uint32_t nowMs, bool live,
                                      float hueRotDeg) {
    if (!live) {
        duckResetAudio(f);
        duckResetTilt(f);
        f.lastUpMs = 0;
        return;
    }

    if (pkt.upMs < f.lastUpMs) duckResetTilt(f);
    f.lastUpMs = pkt.upMs;

    f.hueRotDeg = hueRotDeg;

    float rms = duckDecodeRms(pkt.rms_mean);

    f.ceilV = fmaxf(DUCK_RMS_CEIL_MIN, f.ceilV * expf(-DUCK_CEIL_LEAK_K * dt));
    if (rms > f.ceilV) f.ceilV = rms;
    duckUpdateFloor(f, rms, dt);

    float effFloor = f.floorV * DUCK_FLOOR_HEADROOM;

    if (rms >= f.floorV * DUCK_GATE_OPEN_RATIO) {
        f.gateOpen = true;
        f.gateHold = DUCK_GATE_HOLD_S;
    } else if (f.gateOpen) {
        if (rms >= effFloor) f.gateHold = DUCK_GATE_HOLD_S;
        else if ((f.gateHold -= dt) <= 0.0f) f.gateOpen = false;
    }

    if (!f.gateOpen || rms < effFloor) {
        f.energy = 0.0f;
    } else {
        float db = 20.0f * log10f(rms / effFloor);
        float dbRange = fmaxf(1.0f, 20.0f * log10f(f.ceilV / effFloor));
        f.energy = duckClampf(db / dbRange, 0.0f, 1.0f);
    }

    f.onsetFast += fminf(1.0f, dt / DUCK_ONSET_FAST_TAU) * (rms - f.onsetFast);
    f.onsetSlow += fminf(1.0f, dt / DUCK_ONSET_SLOW_TAU) * (rms - f.onsetSlow);
    float rise = f.onsetFast - f.onsetSlow;
    float thresh = fmaxf(DUCK_ONSET_FLOOR, DUCK_ONSET_RISE_FRAC * f.onsetSlow);
    f.onsetLock = fmaxf(0.0f, f.onsetLock - dt);
    if (f.gateOpen && f.onsetFast > effFloor && rise > thresh
        && f.onsetLock <= 0.0f) {
        f.onsetLock = DUCK_ONSET_REFRAC;
        f.onset = duckClampf(rise / fmaxf(f.onsetSlow, 1.0f), 0.0f, 1.0f);
    } else {
        f.onset = 0.0f;
    }

    if (pkt.flags & DUCK_FLAG_NO_IMU) {
        f.tilt = 0.0f;
        return;
    }

    float ax = (float)pkt.ax, ay = (float)pkt.ay, az = (float)pkt.az;
    float mag = sqrtf(ax * ax + ay * ay + az * az);
    if (mag < 1.0f) return;
    ax /= mag; ay /= mag; az /= mag;

    if (!f.calibrated) {
        if (f.calStartMs == 0) f.calStartMs = nowMs;
        if (pkt.seq != f.calSeq) {
            f.calSeq = pkt.seq;
            f.calSumAx += ax; f.calSumAy += ay; f.calSumAz += az;
            f.calSamples++;
        }
        if (nowMs - f.calStartMs >= DUCK_CAL_MS && f.calSamples > 0) {
            float rx = f.calSumAx / f.calSamples;
            float ry = f.calSumAy / f.calSamples;
            float rz = f.calSumAz / f.calSamples;
            float rm = sqrtf(rx * rx + ry * ry + rz * rz);
            if (rm > 0.1f) {
                f.restAx = rx / rm; f.restAy = ry / rm; f.restAz = rz / rm;
                f.calibrated = true;
            } else {
                duckResetTilt(f);
            }
        }
        f.tilt = 0.0f;
        return;
    }

    float dot = duckClampf(f.restAx * ax + f.restAy * ay + f.restAz * az,
                           -1.0f, 1.0f);
    float angleDeg = acosf(dot) * 180.0f / (float)M_PI;
    if (angleDeg > DUCK_TILT_DEAD_DEG) {
        float hueFrac = duckClampf((angleDeg - DUCK_TILT_DEAD_DEG)
                                   / (DUCK_HUE_MAX_DEG - DUCK_TILT_DEAD_DEG),
                                   0.0f, 1.0f);
        f.hueIdx = hueFrac * 255.0f;
        f.tilt = duckClampf((angleDeg - DUCK_TILT_DEAD_DEG)
                            / DUCK_BLEND_RANGE_DEG, 0.0f, 1.0f);
    } else {
        f.tilt = 0.0f;
    }
}

static inline float duckWaterfallInject(const DuckFeatures &f, bool live) {
    if (!live) return 0.0f;
    return fmaxf(DUCK_WF_IDLE_LEVEL, f.energy) + DUCK_WF_ONSET_GAIN * f.onset;
}

// One strip's advection + paint. lvl/hue/tlt are the caller's persistent
// per-cell state (len entries each); out is the caller's composed frame for
// this strip in 0..255 linear, max-blended into. rnd() must return [0,1).
// reverse = the strip's physical head is its last index.
static inline void duckWaterfallStep(float *lvl, float *hue, float *tlt, int len,
                                     const DuckFeatures &f, float inject, float dt,
                                     float (*rnd)(), float (*out)[3], bool reverse) {
    const float shift = DUCK_WF_SPEED_NORM * (float)len * dt;
    const float decay = expf(-DUCK_WF_DECAY_K * dt);

    // Scratch, not in-place: one frame's advection can span several cells
    // (a 297-LED strip shifts ~1.5 px/frame at 60 Hz), so a cell's source may
    // already have been overwritten.
    static float nlv[DUCK_WF_MAX_LEN], nhu[DUCK_WF_MAX_LEN], ntl[DUCK_WF_MAX_LEN];
    if (len > DUCK_WF_MAX_LEN) len = DUCK_WF_MAX_LEN;

    float cacheHue = -1.0f, cacheR = 0.0f, cacheG = 0.0f, cacheB = 0.0f;
    const uint8_t baseIdx = duckOklchIdx(hueWrap360(f.hueRotDeg));

    for (int i = 0; i < len; i++) {
        float src = (float)i - shift;
        float nl, nh, nt;
        if (src < 0.0f) {
            nl = inject * (1.0f - DUCK_WF_GRAIN + DUCK_WF_GRAIN * 2.0f * rnd());
            nh = f.hueIdx;
            nt = f.tilt;
        } else {
            int i0 = (int)src;
            int i1 = (i0 + 1 < len) ? i0 + 1 : i0;
            float fr = src - (float)i0;
            nl = duckLerpf(lvl[i0], lvl[i1], fr) * decay;
            int n = (fr < 0.5f) ? i0 : i1;
            nh = hue[n];
            nt = tlt[n];
        }
        nlv[i] = nl; nhu[i] = nh; ntl[i] = nt;

        float sp = fminf(nl, 1.0f);
        if (sp < DUCK_WF_DEADBAND) continue;

        // Base water is the arc's own base hue, whitened as the level rises
        // (the crest). It must not be a fixed color outside the arc: additive
        // compositing only preserves hue within a convex arc, so an off-arc
        // base mixed with an arc blue can land in the red band.
        const float crest = 0.35f * sp;
        float colR = duckLerpf((float)oklchConstL[baseIdx][0], 255.0f, crest);
        float colG = duckLerpf((float)oklchConstL[baseIdx][1], 255.0f, crest);
        float colB = duckLerpf((float)oklchConstL[baseIdx][2], 255.0f, crest);
        if (nt > 0.0f) {
            if (nh != cacheHue) {
                cacheHue = nh;
                uint8_t hi = duckOklchIdx(hueWrap360(cacheHue * (360.0f / 255.0f) + f.hueRotDeg));
                // Constant-L, not the variable-L LUT the original duck uses:
                // that one banks on an RGBW white channel carrying luminance,
                // and on RGB-only strips its per-hue lightness swing (red 0.52,
                // purple 0.38) makes the rotation read as dimming. Here the
                // water level owns brightness and hue must not touch it.
                cacheR = (float)oklchConstL[hi][0];
                cacheG = (float)oklchConstL[hi][1];
                cacheB = (float)oklchConstL[hi][2];
            }
            colR = duckLerpf(colR, cacheR, nt);
            colG = duckLerpf(colG, cacheG, nt);
            colB = duckLerpf(colB, cacheB, nt);
        }

        int p = reverse ? (len - 1 - i) : i;
        out[p][0] = fmaxf(out[p][0], colR * sp);
        out[p][1] = fmaxf(out[p][1], colG * sp);
        out[p][2] = fmaxf(out[p][2], colB * sp);
    }

    memcpy(lvl, nlv, sizeof(float) * len);
    memcpy(hue, nhu, sizeof(float) * len);
    memcpy(tlt, ntl, sizeof(float) * len);
}

#endif
