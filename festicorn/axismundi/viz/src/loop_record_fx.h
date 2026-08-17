// Viz twin of src/loop_record.h (firmware): kettle knob-push record/loop.
// Same states, timings, rate law, and POWER.md-capped flash levels; differs
// only in capture format (full-res RGBf at 20 fps — WASM has the memory the
// ESP32 doesn't). Disable via AX_LOOP_RECORD 0 in axismundi_fx.h.
#ifndef LOOP_RECORD_FX_H
#define LOOP_RECORD_FX_H

#define AXLR_CAP_HZ      20
#define AXLR_MAX_FRAMES  160
#define AXLR_RATE_MIN    0.25f
#define AXLR_RATE_MAX    4.0f
#define AXLR_FLASH_RED   (160.0f / 255.0f)
#define AXLR_FLASH_WHITE (52.0f / 255.0f)
#define AXLR_BTN_LOCK_MS 300.0f

enum AxLrState : uint8_t { AXLR_AMBIENT, AXLR_FLASH_REC, AXLR_RECORDING,
                           AXLR_FLASH_PLAY, AXLR_PLAYING, AXLR_FLASH_CLEAR };

static AxLrState axlr_state = AXLR_AMBIENT;
static float    axlr_ms = 0.0f;
static float    axlr_state_ms = 0.0f;
static uint8_t  axlr_buf[AXLR_MAX_FRAMES * HC_NUM_LEDS * 3];
static uint16_t axlr_count = 0, axlr_write = 0;
static float    axlr_play_pos = 0.0f;
static float    axlr_play_dir = 1.0f;
static float    axlr_rate = 1.0f;
static float    axlr_last_cap_ms = -1.0f;
static uint8_t  axlr_prev_btn = 0;
static float    axlr_last_edge_ms = -1e9f;
static int32_t  axlr_prev_enc = 0;
static uint32_t axlr_prev_up = 0;
static bool     axlr_enc_init = false;

static inline bool axlr_knob_diverted() {
    return axlr_state == AXLR_FLASH_PLAY || axlr_state == AXLR_PLAYING;
}

static inline uint8_t* axlr_at(uint16_t f) {
    return axlr_buf + (uint32_t)f * HC_NUM_LEDS * 3;
}

static inline void axlr_enter(AxLrState st) {
    axlr_state = st;
    axlr_state_ms = axlr_ms;
}

static void axlr_button(uint8_t btnBits) {
    uint8_t cur = (btnBits >> KETTLE_BTN_HOLD) & 1;
    bool edge = cur && !axlr_prev_btn;
    axlr_prev_btn = cur;
    if (!edge || axlr_ms - axlr_last_edge_ms < AXLR_BTN_LOCK_MS) return;
    axlr_last_edge_ms = axlr_ms;
    switch (axlr_state) {
        case AXLR_AMBIENT:
            axlr_count = 0; axlr_write = 0; axlr_rate = 1.0f;
            axlr_last_cap_ms = -1.0f;
            axlr_enter(AXLR_FLASH_REC);
            break;
        case AXLR_FLASH_REC:
        case AXLR_RECORDING:
            axlr_enter(AXLR_FLASH_PLAY);
            break;
        case AXLR_FLASH_PLAY:
        case AXLR_PLAYING:
            axlr_enter(AXLR_FLASH_CLEAR);
            break;
        case AXLR_FLASH_CLEAR:
            break;
    }
}

static void axlr_enc_feed(KettleState &k, int32_t enc, uint32_t upMs) {
    if (!axlr_enc_init || upMs < axlr_prev_up) {
        axlr_prev_enc = enc;
        axlr_enc_init = true;
    }
    int32_t d = enc - axlr_prev_enc;
    axlr_prev_enc = enc;
    axlr_prev_up = upMs;
    if (!axlr_knob_diverted() || d == 0) return;
    float rev = (float)d
        / (float)(KETTLE_COUNTS_PER_DETENT * KETTLE_DETENTS_PER_REV);
    k.rotPending -= rev;
    axlr_rate = fminf(AXLR_RATE_MAX,
                      fmaxf(AXLR_RATE_MIN, axlr_rate * powf(2.0f, rev)));
}

static bool axlr_cap_this_frame = false;

static void axlr_tick(float dt) {
    axlr_ms += dt * 1000.0f;
    axlr_cap_this_frame = false;
    float t = axlr_ms - axlr_state_ms;
    switch (axlr_state) {
        case AXLR_FLASH_REC:
            if (t >= 360.0f) {
                axlr_enter(AXLR_RECORDING);
                axlr_last_cap_ms = -1.0f;
            }
            break;
        case AXLR_RECORDING:
            if (axlr_last_cap_ms < 0.0f
                || axlr_ms - axlr_last_cap_ms >= 1000.0f / AXLR_CAP_HZ) {
                axlr_last_cap_ms = axlr_ms;
                axlr_cap_this_frame = true;
            }
            break;
        case AXLR_FLASH_PLAY:
            if (t >= 520.0f) {
                if (axlr_count == 0) axlr_enter(AXLR_FLASH_CLEAR);
                else {
                    axlr_play_pos = 0.0f;
                    axlr_play_dir = 1.0f;
                    axlr_enter(AXLR_PLAYING);
                }
            }
            break;
        case AXLR_PLAYING: {
            float span = (float)axlr_count - 1.0f;
            axlr_play_pos += axlr_play_dir * axlr_rate * (float)AXLR_CAP_HZ * dt;
            if (span <= 0.0f) {
                axlr_play_pos = 0.0f;
            } else {
                while (axlr_play_pos > span || axlr_play_pos < 0.0f) {
                    if (axlr_play_pos > span) {
                        axlr_play_pos = 2.0f * span - axlr_play_pos;
                        axlr_play_dir = -1.0f;
                    }
                    if (axlr_play_pos < 0.0f) {
                        axlr_play_pos = -axlr_play_pos;
                        axlr_play_dir = 1.0f;
                    }
                }
            }
        } break;
        case AXLR_FLASH_CLEAR:
            if (t >= 900.0f) {
                axlr_count = 0;
                axlr_enter(AXLR_AMBIENT);
            }
            break;
        default:
            break;
    }
}

static void axlr_apply(RGBf* out) {
    if (axlr_cap_this_frame) {
        uint8_t *p = axlr_at(axlr_write);
        for (int i = 0; i < HC_NUM_LEDS; i++) {
            p[i * 3 + 0] = (uint8_t)(fminf(fmaxf(out[i].r, 0.0f), 1.0f) * 255.0f);
            p[i * 3 + 1] = (uint8_t)(fminf(fmaxf(out[i].g, 0.0f), 1.0f) * 255.0f);
            p[i * 3 + 2] = (uint8_t)(fminf(fmaxf(out[i].b, 0.0f), 1.0f) * 255.0f);
        }
        axlr_write = (uint16_t)((axlr_write + 1) % AXLR_MAX_FRAMES);
        if (axlr_count < AXLR_MAX_FRAMES) axlr_count++;
    }
    float t = axlr_ms - axlr_state_ms;
    float fr, fg, fb;
    switch (axlr_state) {
        case AXLR_FLASH_REC: {
            bool on = (t < 100.0f) || (t >= 180.0f && t < 280.0f);
            fr = on ? AXLR_FLASH_RED : 0.0f; fg = 0.0f; fb = 0.0f;
        } break;
        case AXLR_FLASH_PLAY: {
            fr = (t < 400.0f) ? AXLR_FLASH_RED : 0.0f; fg = 0.0f; fb = 0.0f;
        } break;
        case AXLR_FLASH_CLEAR: {
            float ph = t / 900.0f;
            float lvl = (ph < 0.5f ? ph : 1.0f - ph) * 2.0f;
            fr = fg = fb = AXLR_FLASH_WHITE * fmaxf(lvl, 0.0f);
        } break;
        case AXLR_PLAYING: {
            uint16_t start = (uint16_t)((axlr_write + AXLR_MAX_FRAMES - axlr_count)
                                        % AXLR_MAX_FRAMES);
            uint16_t f0 = (uint16_t)axlr_play_pos;
            if (f0 >= axlr_count) f0 = axlr_count - 1;
            float ft = axlr_play_pos - (float)f0;
            uint16_t f1 = (uint16_t)((f0 + 1 < axlr_count) ? f0 + 1 : f0);
            const uint8_t *pa = axlr_at((uint16_t)((start + f0) % AXLR_MAX_FRAMES));
            const uint8_t *pb = axlr_at((uint16_t)((start + f1) % AXLR_MAX_FRAMES));
            for (int i = 0; i < HC_NUM_LEDS; i++) {
                out[i].r = ((float)pa[i*3+0] + ((float)pb[i*3+0] - (float)pa[i*3+0]) * ft) / 255.0f;
                out[i].g = ((float)pa[i*3+1] + ((float)pb[i*3+1] - (float)pa[i*3+1]) * ft) / 255.0f;
                out[i].b = ((float)pa[i*3+2] + ((float)pb[i*3+2] - (float)pa[i*3+2]) * ft) / 255.0f;
            }
            return;
        }
        default:
            return;
    }
    for (int i = 0; i < HC_NUM_LEDS; i++) {
        out[i].r = fr; out[i].g = fg; out[i].b = fb;
    }
}

#endif
