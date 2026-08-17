// Kettle knob-push record/loop experiment (2026-07-31). Firmware-only — the
// 3D viz engine does not implement it. Disable via LOOP_RECORD 0 in
// axismundi.cpp. Included mid-file: needs setPixelScaled/lerpf/STRIP_LEN
// above it and is used inside outputRotated below it.
#ifndef LOOP_RECORD_H
#define LOOP_RECORD_H

#define LR_CAP_HZ       20
#define LR_MAX_FRAMES   160
#define LR_SAMPLES      75
#define LR_RATE_MIN     0.25f
#define LR_RATE_MAX     4.0f
// Full-field levels sized to POWER.md's 12.5 mA/LED cap.
#define LR_FLASH_RED    160.0f
#define LR_FLASH_WHITE  52.0f
#define LR_BTN_LOCK_MS  300

enum LrState : uint8_t { LR_AMBIENT, LR_FLASH_REC, LR_RECORDING,
                         LR_FLASH_PLAY, LR_PLAYING, LR_FLASH_CLEAR };

static LrState  lrState = LR_AMBIENT;
static uint32_t lrStateMs = 0;
static uint8_t *lrBuf = nullptr;
static uint16_t lrBufFrames = 0;
static uint16_t lrCount = 0, lrWrite = 0;
static float    lrPlayPos = 0.0f;
static float    lrPlayDir = 1.0f;
static float    lrRate = 1.0f;
static bool     lrCapThisFrame = false;
static uint32_t lrLastCapMs = 0;
static uint8_t  lrPrevBtn = 0;
static uint32_t lrLastEdgeMs = 0;
static int32_t  lrPrevEnc = 0;
static uint32_t lrPrevUp = 0;
static bool     lrEncInit = false;

static inline bool lrKnobDiverted() {
    return lrState == LR_FLASH_PLAY || lrState == LR_PLAYING;
}

static inline uint8_t *lrAt(uint16_t f, uint8_t s) {
    return lrBuf + (((uint32_t)f * NUM_STRIPS + s) * LR_SAMPLES) * 3;
}

static bool lrEnsureBuf() {
    uint16_t want = LR_MAX_FRAMES;
    while (!lrBuf && want >= 40) {
        lrBuf = (uint8_t *)malloc((uint32_t)want * NUM_STRIPS * LR_SAMPLES * 3);
        if (lrBuf) lrBufFrames = want;
        else want /= 2;
    }
    return lrBuf != nullptr;
}

static inline void lrEnter(LrState st, uint32_t nowMs) {
    lrState = st;
    lrStateMs = nowMs;
}

static void lrButton(uint8_t btnBits, uint32_t nowMs) {
    uint8_t cur = (btnBits >> KETTLE_BTN_HOLD) & 1;
    bool edge = cur && !lrPrevBtn;
    lrPrevBtn = cur;
    if (!edge || nowMs - lrLastEdgeMs < LR_BTN_LOCK_MS) return;
    lrLastEdgeMs = nowMs;
    switch (lrState) {
        case LR_AMBIENT:
            if (!lrEnsureBuf()) {
                Serial.println("[loop] buffer alloc failed");
                return;
            }
            lrCount = 0; lrWrite = 0; lrRate = 1.0f; lrLastCapMs = 0;
            lrEnter(LR_FLASH_REC, nowMs);
            Serial.printf("[loop] record (cap %u frames)\n", lrBufFrames);
            break;
        case LR_FLASH_REC:
        case LR_RECORDING:
            lrEnter(LR_FLASH_PLAY, nowMs);
            Serial.printf("[loop] play %u frames rate=%.2f\n", lrCount, lrRate);
            break;
        case LR_FLASH_PLAY:
        case LR_PLAYING:
            lrEnter(LR_FLASH_CLEAR, nowMs);
            Serial.println("[loop] clear");
            break;
        case LR_FLASH_CLEAR:
            break;
    }
}

// Runs after kettleInput has already integrated the encoder into rotPending;
// while the mode is active the spin is backed out and steers rate instead.
static void lrEncFeed(KettleState &k, int32_t enc, uint32_t upMs) {
    if (!lrEncInit || upMs < lrPrevUp) {
        lrPrevEnc = enc;
        lrEncInit = true;
    }
    int32_t d = enc - lrPrevEnc;
    lrPrevEnc = enc;
    lrPrevUp = upMs;
    if (!lrKnobDiverted() || d == 0) return;
    float rev = (float)d
        / (float)(KETTLE_COUNTS_PER_DETENT * KETTLE_DETENTS_PER_REV);
    k.rotPending -= rev;
    lrRate = fminf(LR_RATE_MAX, fmaxf(LR_RATE_MIN, lrRate * powf(2.0f, rev)));
}

static void lrTick(uint32_t nowMs, float dt) {
    lrCapThisFrame = false;
    switch (lrState) {
        case LR_FLASH_REC:
            if (nowMs - lrStateMs >= 360) {
                lrEnter(LR_RECORDING, nowMs);
                lrLastCapMs = 0;
            }
            break;
        case LR_RECORDING:
            if (lrLastCapMs == 0 || nowMs - lrLastCapMs >= 1000 / LR_CAP_HZ) {
                lrLastCapMs = nowMs;
                lrCapThisFrame = true;
            }
            break;
        case LR_FLASH_PLAY:
            if (nowMs - lrStateMs >= 520) {
                if (lrCount == 0) lrEnter(LR_FLASH_CLEAR, nowMs);
                else { lrPlayPos = 0.0f; lrPlayDir = 1.0f; lrEnter(LR_PLAYING, nowMs); }
            }
            break;
        case LR_PLAYING: {
            float span = (float)lrCount - 1.0f;
            lrPlayPos += lrPlayDir * lrRate * (float)LR_CAP_HZ * dt;
            if (span <= 0.0f) {
                lrPlayPos = 0.0f;
            } else {
                while (lrPlayPos > span || lrPlayPos < 0.0f) {
                    if (lrPlayPos > span) {
                        lrPlayPos = 2.0f * span - lrPlayPos;
                        lrPlayDir = -1.0f;
                    }
                    if (lrPlayPos < 0.0f) {
                        lrPlayPos = -lrPlayPos;
                        lrPlayDir = 1.0f;
                    }
                }
            }
        } break;
        case LR_FLASH_CLEAR:
            if (nowMs - lrStateMs >= 900) {
                lrCount = 0;
                lrEnter(LR_AMBIENT, nowMs);
            }
            break;
        default:
            break;
    }
}

static inline void lrTap(uint8_t s, uint16_t i, float r, float g, float b) {
    if (!lrCapThisFrame || (i & 1)) return;
    uint8_t *p = lrAt(lrWrite, s) + (uint16_t)(i >> 1) * 3;
    p[0] = (uint8_t)fminf(r, 255.0f);
    p[1] = (uint8_t)fminf(g, 255.0f);
    p[2] = (uint8_t)fminf(b, 255.0f);
}

static void lrApply(uint32_t nowMs) {
    if (lrCapThisFrame) {
        lrWrite = (uint16_t)((lrWrite + 1) % lrBufFrames);
        if (lrCount < lrBufFrames) lrCount++;
    }
    uint32_t t = nowMs - lrStateMs;
    float fr, fg, fb;
    switch (lrState) {
        case LR_FLASH_REC: {
            bool on = (t < 100) || (t >= 180 && t < 280);
            fr = on ? LR_FLASH_RED : 0.0f; fg = 0.0f; fb = 0.0f;
        } break;
        case LR_FLASH_PLAY: {
            fr = (t < 400) ? LR_FLASH_RED : 0.0f; fg = 0.0f; fb = 0.0f;
        } break;
        case LR_FLASH_CLEAR: {
            float ph = (float)t / 900.0f;
            float lvl = (ph < 0.5f ? ph : 1.0f - ph) * 2.0f;
            fr = fg = fb = LR_FLASH_WHITE * fmaxf(lvl, 0.0f);
        } break;
        case LR_PLAYING: {
            uint16_t start =
                (uint16_t)((lrWrite + lrBufFrames - lrCount) % lrBufFrames);
            uint16_t f0 = (uint16_t)lrPlayPos;
            if (f0 >= lrCount) f0 = lrCount - 1;
            float ft = lrPlayPos - (float)f0;
            uint16_t f1 = (uint16_t)((f0 + 1 < lrCount) ? f0 + 1 : f0);
            uint16_t fa = (uint16_t)((start + f0) % lrBufFrames);
            uint16_t fb2 = (uint16_t)((start + f1) % lrBufFrames);
            for (uint8_t s = 0; s < NUM_STRIPS; s++) {
                const uint8_t *pa = lrAt(fa, s);
                const uint8_t *pb = lrAt(fb2, s);
                uint16_t len = STRIP_LEN[s];
                for (uint16_t i = 0; i < len; i++) {
                    uint16_t j0 = (uint16_t)(i >> 1);
                    uint16_t j1 = (uint16_t)((j0 + 1 < LR_SAMPLES) ? j0 + 1 : j0);
                    float sf = (i & 1) ? 0.5f : 0.0f;
                    float cr = lerpf(lerpf(pa[j0 * 3 + 0], pa[j1 * 3 + 0], sf),
                                     lerpf(pb[j0 * 3 + 0], pb[j1 * 3 + 0], sf), ft);
                    float cg = lerpf(lerpf(pa[j0 * 3 + 1], pa[j1 * 3 + 1], sf),
                                     lerpf(pb[j0 * 3 + 1], pb[j1 * 3 + 1], sf), ft);
                    float cb = lerpf(lerpf(pa[j0 * 3 + 2], pa[j1 * 3 + 2], sf),
                                     lerpf(pb[j0 * 3 + 2], pb[j1 * 3 + 2], sf), ft);
                    setPixelScaled(s, i, cr, cg, cb);
                }
            }
            return;
        }
        default:
            return;
    }
    for (uint8_t s = 0; s < NUM_STRIPS; s++)
        for (uint16_t i = 0; i < STRIP_LEN[s]; i++)
            setPixelScaled(s, i, fr, fg, fb);
}

#endif
